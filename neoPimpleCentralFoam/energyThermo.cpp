// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#include "blocks.hpp"

#include "totalTemperatureFvPatchScalarField.H"
#include "mixedFvPatchFields.H"

#include <set>

namespace hcs
{

namespace
{

/// dst[i] = f(src[i]) for all i; f must be a device-callable (NEON_LAMBDA) closure.
template<typename F>
void mapPointwise(NeoN::Vector<scalar>& dst, const NeoN::Vector<scalar>& src, F f)
{
    auto [dv, sv] = NeoN::views(dst, src);
    NeoN::parallelFor(
        dst.exec(),
        {0, static_cast<NeoN::localIdx>(dv.size())},
        NEON_LAMBDA(const NeoN::localIdx i) { dv[i] = f(sv[i]); },
        "hcs::mapPointwise"
    );
}

/// psi with calculated boundaries, stored in the field database: (7) needs its old time.
VolScalar& createCompressibility(nf::RunTime& rt, fvcc::VectorCollection& fields)
{
    Foam::volScalarField ofPsi(
        Foam::IOobject(
            "psi",
            rt.mesh.time().timeName(),
            rt.mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        rt.mesh,
        Foam::dimensionedScalar(Foam::sqr(Foam::dimTime) / Foam::dimArea, 0),
        Foam::word("calculated")
    );
    return nf::constructAndRegister(fields, rt, ofPsi, false);
}

} // namespace

EnergyThermo::EnergyThermo(
    nf::RunTime& rt,
    fvcc::VectorCollection& fields,
    const PerfectGas& gas,
    const fvcc::BoundaryContext& ctx
)
    : rt_(rt),
      gas_(gas),
      ctx_(ctx),
      ofT_(
          Foam::IOobject(
              "T",
              rt.mesh.time().timeName(),
              rt.mesh,
              Foam::IOobject::MUST_READ,
              Foam::IOobject::NO_WRITE
          ),
          rt.mesh
      ),
      T_(nf::constructFrom(rt.exec, rt.nfMesh, ofT_)),
      h_(createEnthalpy(rt, fields, ofT_, gas)),
      psi_(createCompressibility(rt, fields)),
      c_(makeVolScalar(rt, "c")),
      gammaField_(makeVolScalar(rt, "gamma")),
      TOwn_(makeSurfScalar(rt, "T_own")),
      TNei_(makeSurfScalar(rt, "T_nei")),
      alphaEff_(makeSurfScalar(rt, "alphaEff")),
      own_(makeDirection(rt, "own", 1.0)),
      nei_(makeDirection(rt, "nei", -1.0)),
      reconstructT_(interpolationScheme(rt, "reconstruct(psi)")),
      solver_(h_, rt)
{
    NeoN::Logging::info("2.3 EnergyThermo: read T, created h = h(T)");
}

VolScalar& EnergyThermo::createEnthalpy(
    nf::RunTime& rt,
    fvcc::VectorCollection& fields,
    const Foam::volScalarField& ofT,
    const PerfectGas& gas
)
{
    // The boundary conditions of h are those of T with every temperature scaled by Cp: with a
    // constant Cp, h = Cp T is linear in T, so a fixed T fixes h, a zero gradient of T is a zero
    // gradient of h, totalTemperature(T0) of T is totalTemperature(Cp T0) of h, and
    // inletOutlet(inletValue) / outletInlet(outletValue) of T are the same BC of h with the
    // value scaled by Cp (the flux switch does not depend on the field).
    static const std::set<Foam::word> supported {
        "fixedValue",
        "zeroGradient",
        "calculated",
        "empty",
        "symmetryPlane",
        "symmetry",
        "processor",
        "totalTemperature",
        "inletOutlet",
        "outletInlet"
    };

    Foam::OStringStream os;
    ofT.boundaryField().writeEntries(os);
    Foam::IStringStream is(os.str());
    Foam::dictionary boundaryDict(is);

    // Replaces the entry 'key' of dict by the field values (written as OpenFOAM writes fields).
    auto setField = [](Foam::dictionary& dict, const Foam::word& key, const Foam::scalarField& f)
    {
        Foam::OStringStream es;
        f.writeEntry(key, es);
        Foam::IStringStream eis(es.str());
        dict.merge(Foam::dictionary(eis));
    };

    const auto& Tbf = ofT.boundaryField();
    forAll(Tbf, patchi)
    {
        const auto& patchT = Tbf[patchi];
        const Foam::word& type = patchT.type();
        if (!supported.count(type))
        {
            FatalErrorInFunction << "Boundary condition '" << type << "' of T on patch "
                                 << patchT.patch().name() << " is not supported for h yet"
                                 << Foam::exit(Foam::FatalError);
        }
        Foam::dictionary& patchDict = boundaryDict.subDict(patchT.patch().name());
        if (patchDict.found("value"))
        {
            setField(patchDict, "value", gas.Cp * patchT);
        }
        if (type == "totalTemperature")
        {
            const auto& totalT =
                Foam::refCast<const Foam::totalTemperatureFvPatchScalarField>(patchT);
            setField(patchDict, "T0", gas.Cp * totalT.T0());
        }
        if (type == "inletOutlet" || type == "outletInlet")
        {
            // Both are mixed BCs whose refValue is the inletValue / outletValue.
            const auto& mixedT = Foam::refCast<const Foam::mixedFvPatchScalarField>(patchT);
            setField(
                patchDict,
                type == "inletOutlet" ? "inletValue" : "outletValue",
                gas.Cp * mixedT.refValue()
            );
        }
    }

    Foam::dictionary fieldDict;
    fieldDict.add("dimensions", Foam::dimEnergy / Foam::dimMass);
    fieldDict.add("boundaryField", boundaryDict);
    {
        // "uniform 0" must be a token stream, not a single word.
        Foam::IStringStream internal("internalField uniform 0;");
        fieldDict.merge(Foam::dictionary(internal));
    }

    Foam::volScalarField ofH(
        Foam::IOobject(
            "h",
            ofT.time().timeName(),
            ofT.mesh(),
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        ofT.mesh(),
        fieldDict
    );

    // h = h(T) = Cp T (PerfectGas::h, reference state T = 0, h = 0).
    ofH.primitiveFieldRef() = gas.Cp * ofT.primitiveField();

    return nf::constructAndRegister(fields, rt, ofH, false);
}

void EnergyThermo::initialize()
{
    // (8), (9). The face values (41)-(46) equal the cell values for constant properties.
    Cv_ = gas_.Cv();
    gamma_ = gas_.gamma();
    NeoN::Logging::info("2.3 EnergyThermo: Cv = {}, gamma = {}", Cv_, gamma_);

    NeoN::fill(alphaEff_.internalVector(), gas_.alphaEff());
    NeoN::fill(alphaEff_.boundaryData().value(), gas_.alphaEff());
    NeoN::fill(gammaField_.internalVector(), gamma_);
    NeoN::fill(gammaField_.boundaryData().value(), gamma_);

    // (29), (15), (39), (40) from the T read at construction.
    computeState(false);

    logTransition("2.3 EnergyThermo", state_, State::updated);
    state_ = State::updated;
}

void EnergyThermo::update(
    const Continuity& continuity,
    const MomentumOperators& /*momentum*/,
    const HybridScheme& hybrid,
    const MechanicalSources& sources
)
{
    auto& rho = readOnlyOperand(continuity.rho());
    auto& phiOwn = readOnlyOperand(hybrid.phiOwn());
    auto& phiNei = readOnlyOperand(hybrid.phiNei());
    auto& deltaK = readOnlyOperand(sources.deltaK());
    auto& deltaP = readOnlyOperand(sources.deltaP());

    // The boundary conditions of h first see the current phi, U and psi.
    h_.correctBoundaryConditions(ctx_);

    // (3): ddt(rho, h) + div(phi_own, h) + div(phi_nei, h) - laplacian(mu/Pr, h) + delta K
    //      = delta p
    nf::PDE<NeoN::scalar> hEqn(
        NeoN::dsl::imp::ddt(rho, h_) + NeoN::dsl::imp::div(phiOwn, h_)
        + NeoN::dsl::imp::div(phiNei, h_) - NeoN::dsl::imp::laplacian(alphaEff_, h_)
        + NeoN::dsl::exp::source(deltaK) - NeoN::dsl::exp::source(deltaP)
    );
    solver_.solve(hEqn);
    h_.correctBoundaryConditions(ctx_);

    // (10), (29), (15), (39), (40).
    computeState(true);

    logTransition("2.3 EnergyThermo", state_, State::updated);
    state_ = State::updated;
}

void EnergyThermo::computeState(bool fromEnthalpy)
{
    // Copies for the device kernels: a lambda must not capture 'this'.
    const PerfectGas gas = gas_;

    if (fromEnthalpy)
    {
        // (10): T = T(h) by Newton iterations, starting from the current T.
        auto [Tv, hv] = NeoN::views(T_.internalVector(), h_.internalVector());
        NeoN::parallelFor(
            T_.exec(),
            {0, static_cast<NeoN::localIdx>(Tv.size())},
            NEON_LAMBDA(const NeoN::localIdx i) { Tv[i] = gas.T(hv[i], Tv[i]); },
            "hcs::TfromH"
        );
        // The boundary conditions of T are consistent with those of h: a fixed T gives the
        // fixed h, and zero gradients of T and h coincide for a constant Cp.
        T_.correctBoundaryConditions(ctx_);
    }

    // (29), (15) in the cells and on the boundary faces.
    auto psiOf = NEON_LAMBDA(const scalar T) { return gas.psi(T); };
    auto cOf = NEON_LAMBDA(const scalar T) { return gas.c(T); };
    mapPointwise(psi_.internalVector(), T_.internalVector(), psiOf);
    mapPointwise(psi_.boundaryData().value(), T_.boundaryData().value(), psiOf);
    mapPointwise(c_.internalVector(), T_.internalVector(), cOf);
    mapPointwise(c_.boundaryData().value(), T_.boundaryData().value(), cOf);

    // (39), (40): T reconstructed from the owner and from the neighbour side.
    reconstructT_.interpolate(own_, T_, TOwn_);
    reconstructT_.interpolate(nei_, T_, TNei_);
}

void EnergyThermo::reset() { state_ = State::outdated; }

void EnergyThermo::advanceTime()
{
    fvcc::rotateOldTimes(h_);
    fvcc::rotateOldTimes(psi_);
}

void EnergyThermo::write() const { nf::write(T_, rt_.mesh); }

// * * * * * * * * * * * * * * * * Predictor * * * * * * * * * * * * * * * * //

Predictor::Predictor(
    nf::RunTime& rt,
    fvcc::VectorCollection& fields,
    const PerfectGas& gas,
    const fvcc::BoundaryContext& ctx
)
    : continuity_(rt, fields), momentum_(rt, fields, gas, ctx), thermo_(rt, fields, gas, ctx)
{}

void Predictor::update(
    bool firstIteration,
    const PressureEquation& pressure,
    const KNP& knp,
    const HybridScheme& hybrid,
    const MechanicalSources& sources
)
{
    if (firstIteration)
    {
        continuity_.update(thermo_, pressure, hybrid);
    }
    momentum_.update(continuity_, pressure, knp, hybrid);
    thermo_.update(continuity_, momentum_, hybrid, sources);
}

void Predictor::reset()
{
    continuity_.reset();
    momentum_.reset();
    thermo_.reset();
}

} // namespace hcs
