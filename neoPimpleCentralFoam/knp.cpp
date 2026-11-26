// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#include "blocks.hpp"

namespace hcs
{

KNP::KNP(nf::RunTime& rt)
    : rt_(rt),
      geometry_(rt),
      cOwn_(makeSurfScalar(rt, "c_own")),
      cNei_(makeSurfScalar(rt, "c_nei")),
      W_(makeVolScalar(rt, "W")),
      alphaOwn_(makeSurfScalar(rt, "alpha_own")),
      alphaNei_(makeSurfScalar(rt, "alpha_nei")),
      aSf_(makeSurfScalar(rt, "aSf")),
      psiOwn_(makeSurfScalar(rt, "psi_own")),
      psiNei_(makeSurfScalar(rt, "psi_nei")),
      phidOwn_(makeSurfScalar(rt, "phid_own")),
      phidNei_(makeSurfScalar(rt, "phid_nei")),
      DpOwn_(makeSurfScalar(rt, "Dp_own")),
      DpNei_(makeSurfScalar(rt, "Dp_nei")),
      wOwn_(makeSurfScalar(rt, "W_own")),
      wNei_(makeSurfScalar(rt, "W_nei")),
      phivOwn_(makeSurfScalar(rt, "phiv_own")),
      phivNei_(makeSurfScalar(rt, "phiv_nei")),
      cSfOwn_(makeSurfScalar(rt, "cSf_own")),
      cSfNei_(makeSurfScalar(rt, "cSf_nei")),
      ap_(makeSurfScalar(rt, "ap")),
      am_(makeSurfScalar(rt, "am")),
      psiLin_(makeSurfScalar(rt, "psif")),
      WU_(makeVolField<NeoN::Vec3>(rt, "WU")),
      WUOwn_(makeSurfField<NeoN::Vec3>(rt, "WU_own")),
      WUNei_(makeSurfField<NeoN::Vec3>(rt, "WU_nei")),
      WrAU_(makeVolScalar(rt, "WrAU")),
      own_(makeDirection(rt, "own", 1.0)),
      nei_(makeDirection(rt, "nei", -1.0)),
      reconstructW_(interpolationScheme(rt, "reconstruct(W)")),
      reconstructU_(interpolationScheme<NeoN::Vec3>(rt, "reconstruct(U)")),
      reconstructPsi_(interpolationScheme(rt, "reconstruct(psi)")),
      reconstructDp_(interpolationScheme(rt, "reconstruct(Dp)")),
      linear_(linearScheme(rt))
{
}

void KNP::initialize(
    const Continuity& continuity,
    const MomentumOperators& momentum,
    const EnergyThermo& thermo,
    const HybridScheme& hybrid
)
{
    compute(continuity, momentum, thermo, hybrid);
}

void KNP::update(
    const Continuity& continuity,
    const MomentumOperators& momentum,
    const EnergyThermo& thermo,
    const HybridScheme& hybrid
)
{
    compute(continuity, momentum, thermo, hybrid);
}

void KNP::reset() { state_ = State::outdated; }

void KNP::volumetricFluxes(const VolVector& velocity)
{
    kernels::fieldwise(
        WU_, NEON_LAMBDA(const scalar w, const NeoN::Vec3& u) { return w * u; }, W_, velocity
    );
    reconstructU_.interpolate(own_, WU_, WUOwn_);
    reconstructU_.interpolate(nei_, WU_, WUNei_);

    auto phiv = NEON_LAMBDA(const NeoN::Vec3& wu, const scalar w, const NeoN::Vec3& sf)
    {
        return (wu * (scalar(1) / w)) & sf;
    };
    kernels::fieldwise(phivOwn_, phiv, WUOwn_, wOwn_, geometry_.Sf);
    kernels::fieldwise(phivNei_, phiv, WUNei_, wNei_, geometry_.Sf);
}

void KNP::compute(
    const Continuity& continuity,
    const MomentumOperators& momentum,
    const EnergyThermo& thermo,
    const HybridScheme& hybrid
)
{
    const PerfectGas gas = thermo.gas();
    const scalar R = gas.R;

    // (37), (38): c from the reconstructed T (constant Cp, Cv, R on the faces).
    auto cOf = NEON_LAMBDA(const scalar T) { return gas.c(T); };
    kernels::fieldwise(cOwn_, cOf, thermo.TOwn());
    kernels::fieldwise(cNei_, cOf, thermo.TNei());

    // (30): W = rho R (p/T).
    kernels::fieldwise(W_, NEON_LAMBDA(const scalar rho) { return rho * R; }, continuity.rho());
    reconstructW_.interpolate(own_, W_, wOwn_);
    reconstructW_.interpolate(nei_, W_, wNei_);

    // (31), (32) from U; (36), (35).
    volumetricFluxes(momentum.U());
    auto cSf = NEON_LAMBDA(const scalar c, const scalar magSf) { return c * magSf; };
    kernels::fieldwise(cSfOwn_, cSf, cOwn_, geometry_.magSf);
    kernels::fieldwise(cSfNei_, cSf, cNei_, geometry_.magSf);
    kernels::fieldwise(
        ap_,
        NEON_LAMBDA(const scalar po, const scalar pn, const scalar co, const scalar cn) {
            return Kokkos::max(Kokkos::max(po + co, pn + cn), scalar(0));
        },
        phivOwn_,
        phivNei_,
        cSfOwn_,
        cSfNei_
    );
    kernels::fieldwise(
        am_,
        NEON_LAMBDA(const scalar po, const scalar pn, const scalar co, const scalar cn) {
            return Kokkos::min(Kokkos::min(po - co, pn - cn), scalar(0));
        },
        phivOwn_,
        phivNei_,
        cSfOwn_,
        cSfNei_
    );

    // (34), (33), (47): Kurganov.
    kernels::fieldwise(
        alphaOwn_,
        NEON_LAMBDA(const scalar ap, const scalar am) { return ap / (ap - am); },
        ap_,
        am_
    );
    kernels::fieldwise(
        aSf_, NEON_LAMBDA(const scalar am, const scalar a) { return am * a; }, am_, alphaOwn_
    );
    kernels::fieldwise(alphaNei_, NEON_LAMBDA(const scalar a) { return scalar(1) - a; }, alphaOwn_);

    // (27), (28): psi_side = kappa psi_side^reconstructed + (1 - kappa) psi^linear.
    // The original reconstructs 1/T and divides by R; with a constant R that is the same as
    // reconstructing psi = 1/(R T), because the limiters are invariant to scaling.
    const auto& psi = thermo.psi();
    linear_.interpolate(psi, psiLin_);
    reconstructPsi_.interpolate(own_, psi, psiOwn_);
    reconstructPsi_.interpolate(nei_, psi, psiNei_);
    auto blend = NEON_LAMBDA(const scalar side, const scalar lin, const scalar kappa)
    {
        return kappa * side + (scalar(1) - kappa) * lin;
    };
    kernels::fieldwise(psiOwn_, blend, psiOwn_, psiLin_, hybrid.kappa());
    kernels::fieldwise(psiNei_, blend, psiNei_, psiLin_, hybrid.kappa());

    // (21), (22): phid_side = (alpha_side phiv_side(H/A) -/+ aSf) psi_side.
    volumetricFluxes(momentum.HbyA());
    kernels::fieldwise(
        phidOwn_,
        NEON_LAMBDA(const scalar a, const scalar phiv, const scalar aSf, const scalar psiS) {
            return (a * phiv - aSf) * psiS;
        },
        alphaOwn_,
        phivOwn_,
        aSf_,
        psiOwn_
    );
    kernels::fieldwise(
        phidNei_,
        NEON_LAMBDA(const scalar a, const scalar phiv, const scalar aSf, const scalar psiS) {
            return (a * phiv + aSf) * psiS;
        },
        alphaNei_,
        phivNei_,
        aSf_,
        psiNei_
    );

    // (23), (24): Dp_side = alpha_side (W/A)_side / R.
    kernels::fieldwise(
        WrAU_, NEON_LAMBDA(const scalar w, const scalar rAU) { return w * rAU; }, W_, momentum.rAU()
    );
    reconstructDp_.interpolate(own_, WrAU_, DpOwn_);
    reconstructDp_.interpolate(nei_, WrAU_, DpNei_);
    auto Dp = NEON_LAMBDA(const scalar a, const scalar wrAU) { return a * wrAU / R; };
    kernels::fieldwise(DpOwn_, Dp, alphaOwn_, DpOwn_);
    kernels::fieldwise(DpNei_, Dp, alphaNei_, DpNei_);

    logTransition("3.1 KNP", state_, State::updated);
    state_ = State::updated;
}

// * * * * * * * * * * * * * * * * * PVC * * * * * * * * * * * * * * * * * //

PVC::PVC(nf::RunTime& rt, fvcc::VectorCollection& fields, const fvcc::BoundaryContext& ctx)
    : knp_(rt), pressure_(rt, fields, ctx)
{}

void PVC::update(
    const Continuity& continuity,
    const MomentumOperators& momentum,
    const EnergyThermo& thermo,
    const HybridScheme& hybrid
)
{
    knp_.update(continuity, momentum, thermo, hybrid);
    pressure_.update(knp_, thermo);
}

void PVC::reset()
{
    knp_.reset();
    pressure_.reset();
}

} // namespace hcs
