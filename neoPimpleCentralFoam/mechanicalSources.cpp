// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#include "blocks.hpp"

namespace hcs
{

namespace
{

VolScalar& createKineticEnergy(nf::RunTime& rt, fvcc::VectorCollection& fields)
{
    Foam::volScalarField ofK(
        Foam::IOobject(
            "K",
            rt.mesh.time().timeName(),
            rt.mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        rt.mesh,
        Foam::dimensionedScalar(Foam::dimVelocity * Foam::dimVelocity, 0),
        Foam::word("calculated")
    );
    return nf::constructAndRegister(fields, rt, ofK, false);
}

/// rho K in the cells.
NeoN::Vector<scalar> rhoK(const VolScalar& rho, const VolScalar& K)
{
    NeoN::Vector<scalar> result(rho.exec(), rho.internalVector().size(), scalar(0));
    kernels::pointwise(
        result,
        NEON_LAMBDA(const scalar r, const scalar k) { return r * k; },
        rho.internalVector(),
        K.internalVector()
    );
    return result;
}

} // namespace

MechanicalSources::MechanicalSources(nf::RunTime& rt, fvcc::VectorCollection& fields)
    : rt_(rt),
      K_(createKineticEnergy(rt, fields)),
      deltaK_(makeVolScalar(rt, "deltaK")),
      deltaP_(makeVolScalar(rt, "deltaP")),
      KfOwn_(makeSurfScalar(rt, "K_own")),
      KfNei_(makeSurfScalar(rt, "K_nei")),
      flux_(makeSurfScalar(rt, "phiK")),
      divKOwn_(divInterpolationScheme(rt, "div(phi_own,K)")),
      divKNei_(divInterpolationScheme(rt, "div(phi_nei,K)")),
      geometry_(rt),
      gradU_(makeVolField<NeoN::Tensor>(rt, "gradU_work")),
      tauU_(makeVolField<NeoN::Vec3>(rt, "tauU")),
      tauUf_(makeSurfField<NeoN::Vec3>(rt, "tauUf")),
      work_(makeSurfScalar(rt, "viscousWork")),
      workCell_(makeVolScalar(rt, "viscousWorkCell")),
      linearV_(rt.exec, rt.nfMesh, NeoN::TokenList({std::string("linear")}))
{}

void MechanicalSources::computeK(const MomentumOperators& momentum)
{
    // (5): K = |U|^2/2 in the cells and on the boundary faces.
    kernels::fieldwise(
        K_, NEON_LAMBDA(const NeoN::Vec3& u) { return scalar(0.5) * (u & u); }, momentum.U()
    );
}

void MechanicalSources::initialize(const MomentumOperators& momentum)
{
    // Initialization step 8: K = U.U/2; delta K and delta p are zero after construction.
    computeK(momentum);
    logTransition("6 MechanicalSources", state_, State::updated);
    state_ = State::updated;
}

void MechanicalSources::update(
    const MomentumOperators& momentum,
    const Continuity& continuity,
    const PressureEquation& pressure,
    const HybridScheme& hybrid
)
{
    computeK(momentum);

    // (4), convective part: div(phi_own, K) + div(phi_nei, K), K interpolated with the schemes
    // of div(phi_own,K), div(phi_nei,K) (fvc::div(phi, K)).
    divKOwn_.interpolate(hybrid.phiOwn(), K_, KfOwn_);
    divKNei_.interpolate(hybrid.phiNei(), K_, KfNei_);
    kernels::fieldwise(
        flux_,
        NEON_LAMBDA(const scalar po, const scalar ko, const scalar pn, const scalar kn) {
            return po * ko + pn * kn;
        },
        hybrid.phiOwn(),
        KfOwn_,
        hybrid.phiNei(),
        KfNei_
    );
    kernels::surfaceIntegrate(flux_, deltaK_);

    // (4), time derivative: + (rho K - rho_0 K_0)/dt.
    const scalar rDt = scalar(1) / rt_.dt;
    const auto& rho = continuity.rho();
    const auto rhoKNew = rhoK(rho, K_);
    const auto rhoKOld = rhoK(fvcc::oldTime(rho), fvcc::oldTime(static_cast<const VolScalar&>(K_)));
    kernels::pointwise(
        deltaK_.internalVector(),
        NEON_LAMBDA(const scalar conv, const scalar rk, const scalar rk0) {
            return conv + (rk - rk0) * rDt;
        },
        deltaK_.internalVector(),
        rhoKNew,
        rhoKOld
    );

    // Viscous work, moved from the right side of (3) into delta K:
    // delta K += div(linear(devRhoReff & U) & Sf) = -div(linear(tau & U) & Sf),
    // tau = mu dev(grad(U) + grad(U)^T).
    const scalar mu = momentum.gas().mu;
    if (mu > 0)
    {
        kernels::velocityGradient(momentum.U(), gradU_);
        kernels::fieldwise(
            tauU_,
            NEON_LAMBDA(const NeoN::Tensor& g, const NeoN::Vec3& u) {
                const NeoN::Tensor s = g + g.T();
                return (mu * (s - NeoN::Tensor(s.trace() / scalar(3)))) & u;
            },
            gradU_,
            momentum.U()
        );
        linearV_.interpolate(tauU_, tauUf_);
        kernels::fieldwise(
            work_,
            NEON_LAMBDA(const NeoN::Vec3& t, const NeoN::Vec3& sf) { return t & sf; },
            tauUf_,
            geometry_.Sf
        );
        kernels::surfaceIntegrate(work_, workCell_);
        kernels::pointwise(
            deltaK_.internalVector(),
            NEON_LAMBDA(const scalar dk, const scalar w) { return dk - w; },
            deltaK_.internalVector(),
            workCell_.internalVector()
        );
    }

    // (6): delta p = (p - p_0)/dt.
    const auto& p = pressure.p();
    kernels::pointwise(
        deltaP_.internalVector(),
        NEON_LAMBDA(const scalar pNew, const scalar pOld) { return (pNew - pOld) * rDt; },
        p.internalVector(),
        fvcc::oldTime(p).internalVector()
    );

    logTransition("6 MechanicalSources", state_, State::updated);
    state_ = State::updated;
}

void MechanicalSources::reset() { state_ = State::outdated; }

void MechanicalSources::advanceTime() { fvcc::rotateOldTimes(K_); }

// * * * * * * * * * * * * * * * * Corrector * * * * * * * * * * * * * * * * //

void Corrector::update(Predictor& predictor, const PVC& pvc, const HybridScheme& hybrid)
{
    // 4.1: second run of 2.2 (the pressure equation is solved, so it corrects U).
    predictor.momentum().update(predictor.continuity(), pvc.pressure(), pvc.knp(), hybrid);
    // 4.2: second run of 2.1 (the pressure equation is solved, so it applies (11)).
    predictor.continuity().update(predictor.thermo(), pvc.pressure(), hybrid);

    logTransition("4 Corrector", state_, State::updated);
    state_ = State::updated;
}

void Corrector::reset() { state_ = State::outdated; }

// * * * * * * * * * * * * * * * EndOfIteration * * * * * * * * * * * * * * * //

EndOfIteration::EndOfIteration(const nf::RunTime& rt) : printStatistics_(printStatistics(rt))
{
    NeoN::Logging::info("printStatistics = {}", printStatistics_);
}

void EndOfIteration::update(
    TimeStep& timeStep,
    Predictor& predictor,
    PVC& pvc,
    Corrector& corrector,
    HybridScheme& hybrid,
    MechanicalSources& sources
)
{
    timeStep.reset();
    predictor.reset();
    pvc.reset();
    corrector.reset();
    hybrid.reset();
    sources.reset();
    NeoN::Logging::info("7 EndOfIteration: all blocks outdated");

    if (!printStatistics_)
    {
        return;
    }
    const auto p = kernels::minMax(pvc.pressure().p().internalVector());
    const auto T = kernels::minMax(predictor.thermo().T().internalVector());
    const auto U = kernels::minMaxMag(predictor.momentum().U().internalVector());
    NeoN::Logging::info("    max/min p: {}/{}", p.max, p.min);
    NeoN::Logging::info("    max/min T: {}/{}", T.max, T.min);
    NeoN::Logging::info("    max/min mag(U): {}/{}", U.max, U.min);
}

} // namespace hcs
