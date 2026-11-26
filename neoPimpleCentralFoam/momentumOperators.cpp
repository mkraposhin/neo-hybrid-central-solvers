// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#include "blocks.hpp"

#include "cellQuality.H"

namespace hcs
{

MomentumOperators::MomentumOperators(
    nf::RunTime& rt,
    fvcc::VectorCollection& fields,
    const PerfectGas& gas,
    const fvcc::BoundaryContext& ctx
)
    : rt_(rt),
      gas_(gas),
      ctx_(ctx),
      ofU_(
          Foam::IOobject(
              "U",
              rt.mesh.time().timeName(),
              rt.mesh,
              Foam::IOobject::MUST_READ,
              Foam::IOobject::NO_WRITE
          ),
          rt.mesh
      ),
      U_(nf::constructAndRegister(fields, rt, ofU_, false)),
      rAU_(makeVolScalar(rt, "rAU")),
      HbyA_(
          rt.exec,
          "HbyA",
          rt.nfMesh,
          U_.internalVector(),
          U_.boundaryData(),
          U_.boundaryConditions()
      ),
      geometry_(rt),
      pf_(makeSurfScalar(rt, "pf")),
      gradp_(makeVolField<NeoN::Vec3>(rt, "gradp")),
      solver_(U_, rt),
      muEff_(makeSurfScalar(rt, "muEff")),
      gradU_(makeVolField<NeoN::Tensor>(rt, "gradU")),
      stress_(makeVolField<NeoN::Tensor>(rt, "viscousStress")),
      viscSource_(makeVolField<NeoN::Vec3>(rt, "viscousSource")),
      badQualityCells_(markBadQualityCells(rt)),
      limGradP_(makeVolField<NeoN::Vec3>(rt, "limGradP"))
{
    NeoN::fill(muEff_.internalVector(), gas_.mu);
    NeoN::fill(muEff_.boundaryData().value(), gas_.mu);
    NeoN::Logging::info("2.2 MomentumOperators: read U");
}

NeoN::Vector<NeoN::localIdx> MomentumOperators::markBadQualityCells(const nf::RunTime& rt)
{
    const auto& pimple = rt.mesh.solutionDict().subDict("PIMPLE");
    const scalar nonOrthogonalityThreshold = pimple.get<Foam::scalar>("nonOrthogonalityThreshold");
    const scalar skewnessThreshold = pimple.get<Foam::scalar>("skewnessThreshold");

    Foam::cellQuality cq(rt.mesh);
    const Foam::scalarField nonOrtho(cq.nonOrthogonality());
    const Foam::scalarField skewness(cq.skewness());

    std::vector<NeoN::localIdx> cells;
    forAll(nonOrtho, celli)
    {
        if (nonOrtho[celli] >= nonOrthogonalityThreshold || skewness[celli] >= skewnessThreshold)
        {
            cells.push_back(static_cast<NeoN::localIdx>(celli));
        }
    }
    NeoN::Logging::info(
        "2.2 MomentumOperators: {} cells of bad quality (nonOrthogonality >= {} or skewness >= {})"
        " use the limited pressure gradient",
        cells.size(),
        nonOrthogonalityThreshold,
        skewnessThreshold
    );
    return NeoN::Vector<NeoN::localIdx>(rt.exec, cells);
}

void MomentumOperators::initialize(const Continuity& continuity)
{
    // As the original (createFields.H): rAU = 1/(rho v_one), v_one = 1 1/s; the cell values are
    // then multiplied by deltaT and divided by V (primitiveFieldRef only). It is replaced by 1/A
    // of UEqn before its first use; H/A = U.
    kernels::fieldwise(
        rAU_, NEON_LAMBDA(const scalar rho) { return scalar(1) / rho; }, continuity.rho()
    );
    const scalar dt = rt_.dt;
    auto [rAUv, volV] = NeoN::views(rAU_.internalVector(), rt_.nfMesh.cellVolumes());
    NeoN::parallelFor(
        rAU_.exec(),
        {0, static_cast<NeoN::localIdx>(rAU_.internalVector().size())},
        NEON_LAMBDA(const NeoN::localIdx i) { rAUv[i] *= dt / volV[i]; },
        "hcs::initialRAU"
    );
    HbyA_.internalVector() = U_.internalVector();
    HbyA_.boundaryData().value() = U_.boundaryData().value();

    logTransition("2.2 MomentumOperators", state_, State::assembled);
    state_ = State::assembled;
}

void MomentumOperators::update(
    const Continuity& continuity,
    const PressureEquation& pressure,
    const KNP& knp,
    const HybridScheme& hybrid
)
{
    if (pressure.state() == PressureEquation::State::solved)
    {
        correct(pressure, knp);
    }
    else
    {
        assemble(continuity, pressure, hybrid);
    }
}

void MomentumOperators::reset() { state_ = State::outdated; }

void MomentumOperators::advanceTime() { fvcc::rotateOldTimes(U_); }

void MomentumOperators::write() const { nf::write(U_, rt_.mesh); }

void MomentumOperators::assemble(
    const Continuity& continuity,
    const PressureEquation& /*pressure*/,
    const HybridScheme& hybrid
)
{
    auto& rho = readOnlyOperand(continuity.rho());
    auto& phiOwn = readOnlyOperand(hybrid.phiOwn());
    auto& phiNei = readOnlyOperand(hybrid.phiNei());

    // (2) without the pressure gradient:
    //   ddt(rho, U) + div(phi_own, U) + div(phi_nei, U) + divDevRhoReff(U),
    //   divDevRhoReff(U) = -laplacian(mu, U) - div(mu dev2(T(grad(U)))).
    // The balance is assembled only.
    auto expr = NeoN::dsl::imp::ddt(rho, U_) + NeoN::dsl::imp::div(phiOwn, U_)
              + NeoN::dsl::imp::div(phiNei, U_);
    if (gas_.mu > 0)
    {
        const scalar mu = gas_.mu;
        kernels::velocityGradient(U_, gradU_);
        // gradU_ is NeoN's dU_i/dx_j, i.e. OpenFOAM's T(grad(U)); dev2(A) = A - 2/3 tr(A) I.
        kernels::fieldwise(
            stress_,
            NEON_LAMBDA(const NeoN::Tensor& g) {
                return mu * (g - NeoN::Tensor(scalar(2) / scalar(3) * g.trace()));
            },
            gradU_
        );
        kernels::divTensor(stress_, geometry_.Sf, viscSource_);
        kernels::pointwise(
            viscSource_.internalVector(),
            NEON_LAMBDA(const NeoN::Vec3& v) { return scalar(-1) * v; },
            viscSource_.internalVector()
        );
        expr.addOperator(scalar(-1) * NeoN::dsl::imp::laplacian(muEff_, U_));
        expr.addOperator(NeoN::dsl::exp::source(viscSource_));
    }
    nf::PDE<NeoN::Vec3> UEqn(expr);
    solver_.assemble(UEqn);

    // (25), (26): rAU = 1/A, HbyA = H/A; HbyA takes the boundary values of U.
    auto [rAU, HbyA] = nf::computeRAUandHByA(UEqn);
    rAU_.internalVector() = rAU.internalVector();
    rAU_.boundaryData().value() = rAU.boundaryData().value();
    // HbyA.boundaryFieldRef() == U.boundaryField() on the physical patches; the processor
    // patches keep the exchanged neighbour values of H/A (OpenFOAM's patchNeighbourField).
    HbyA_.internalVector() = HbyA.internalVector();
    HbyA_.boundaryData().value() = HbyA.boundaryData().value();
    kernels::copyRange(
        HbyA_.boundaryData().value(), U_.boundaryData().value(), 0, rt_.nfMesh.nBoundaryFaces()
    );

    logTransition("2.2 MomentumOperators", state_, State::assembled);
    state_ = State::assembled;
}

void MomentumOperators::correct(const PressureEquation& pressure, const KNP& knp)
{
    // (48): U = H/A - 1/A div((alpha_o p_o + alpha_n p_n) Sf).
    kernels::fieldwise(
        pf_,
        NEON_LAMBDA(const scalar ao, const scalar po, const scalar an, const scalar pn) {
            return ao * po + an * pn;
        },
        knp.alphaOwn(),
        pressure.pOwn(),
        knp.alphaNei(),
        pressure.pNei()
    );
    kernels::divScalarSf(pf_, geometry_.Sf, gradp_);
    if (badQualityCells_.size() > 0)
    {
        // limitPressureGradient.H
        fvcc::GaussGreenGrad(U_.exec(), U_.mesh()).grad(pressure.p(), limGradP_);
        kernels::copyAt(gradp_.internalVector(), limGradP_.internalVector(), badQualityCells_);
    }

    kernels::pointwise(
        U_.internalVector(),
        NEON_LAMBDA(const NeoN::Vec3& hbya, const scalar rAU, const NeoN::Vec3& gp) {
            return hbya - rAU * gp;
        },
        HbyA_.internalVector(),
        rAU_.internalVector(),
        gradp_.internalVector()
    );
    U_.correctBoundaryConditions(ctx_);

    logTransition("2.2 MomentumOperators", state_, State::corrected);
    state_ = State::corrected;
}

} // namespace hcs
