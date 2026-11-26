// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#include "blocks.hpp"

namespace hcs
{

namespace
{

const Foam::dictionary& pimpleDict(const nf::RunTime& rt)
{
    return rt.mesh.solutionDict().subDict("PIMPLE");
}

// The boundary data holds the physical boundary faces [0, nBoundaryFaces) followed by the
// processor faces [nBoundaryFaces, nBoundaryFaces + nProcBoundaryFaces).

/// kappa = value on the boundary faces [begin, end).
void fillKappa(SurfScalar& kappa, const NeoN::localIdx begin, const NeoN::localIdx end, const scalar value)
{
    if (end <= begin) return;
    auto kV = kappa.boundaryData().value().view();
    NeoN::parallelFor(
        kappa.exec(), {begin, end}, NEON_LAMBDA(const NeoN::localIdx f) { kV[f] = value; }, "hcs::fillKappa"
    );
}

/// kappa = 0 on the physical boundary faces.
///
/// A boundary face has only one state (the cell's); the neighbour state of the Riemann problem
/// that KNP (kappa = 1) solves is missing. Until a proper boundary Riemann solver is introduced,
/// the numerical model is simplified there: KNP is switched off (pure PIMPLE flux, kappa = 0),
/// as in the original. This choice is debatable: a boundary condition that defines the outer
/// state (e.g. a supersonic inlet) could allow kappa > 0.
void zeroPhysicalBoundaryKappa(const NeoN::UnstructuredMesh& mesh, SurfScalar& kappa)
{
    fillKappa(kappa, 0, mesh.nBoundaryFaces(), scalar(0));
}

/// (13), kappa = min(Ma/ACo, 1), on the processor faces. A processor face is an internal face
/// split between two ranks: both states exist (the ghost value is the neighbour cell's value),
/// so the Riemann problem is the same as on an internal face, and kappa must not depend on the
/// decomposition.
void computeProcKappa(
    const NeoN::UnstructuredMesh& mesh, const SurfScalar& Ma, const SurfScalar& ACo, SurfScalar& kappa
)
{
    const NeoN::localIdx begin = mesh.nBoundaryFaces();
    const NeoN::localIdx end = begin + mesh.nProcBoundaryFaces();
    if (end <= begin) return;
    auto [kV, maV, acoV] =
        NeoN::views(kappa.boundaryData().value(), Ma.boundaryData().value(), ACo.boundaryData().value());
    NeoN::parallelFor(
        kappa.exec(),
        {begin, end},
        NEON_LAMBDA(const NeoN::localIdx f) { kV[f] = Kokkos::min(maV[f] / acoV[f], scalar(1)); },
        "hcs::procKappa"
    );
}

/// As the original kappaFunction::resetCoupledBoundaries: kappa = 0 on the coupled faces (here:
/// the processor faces, the only coupled faces NeoN has).
void resetCoupledBoundaries(const NeoN::UnstructuredMesh& mesh, SurfScalar& kappa)
{
    const NeoN::localIdx begin = mesh.nBoundaryFaces();
    fillKappa(kappa, begin, begin + mesh.nProcBoundaryFaces(), scalar(0));
}

} // namespace

HybridScheme::HybridScheme(nf::RunTime& rt)
    : rt_(rt),
      kappa0_(pimpleDict(rt).getOrDefault<Foam::scalar>("kappa0", 0)),
      printStatistics_(printStatistics(rt)),
      geometry_(rt),
      deltaCoeffs_(makeSurfScalar(rt, "deltaCoeffs")),
      amaxSmall_(0),
      kappa_(makeSurfScalar(rt, "kappa")),
      phiOwn_(makeSurfScalar(rt, "phi_own")),
      phiNei_(makeSurfScalar(rt, "phi_nei")),
      phi_(makeSurfScalar(rt, "phi")),
      amaxSf_(makeSurfScalar(rt, "amaxSf")),
      ACo_(makeSurfScalar(rt, "FaceAcCourant")),
      Ma_(makeSurfScalar(rt, "Maf")),
      rhoA_(makeSurfScalar(rt, "rhoA")),
      cA_(makeSurfScalar(rt, "cA"))
{
    if (kappa0_ < 0 || kappa0_ > 1)
    {
        FatalIOErrorInFunction(pimpleDict(rt))
            << "kappa0 = " << kappa0_ << " is outside [0, 1]" << Foam::exit(Foam::FatalIOError);
    }
    kernels::deltaCoeffs(rt.nfMesh, deltaCoeffs_);
    amaxSmall_ = Foam::SMALL * kernels::minMax(geometry_.magSf.internalVector()).min;
    // (13) is always OpenFOAM's standardMachToAcCourantRatio: kappa = min(Ma/ACo, 1).
    NeoN::Logging::info("5 HybridScheme: kappa0 = {}, kappa = min(Ma/ACo, 1)", kappa0_);
}

void HybridScheme::initialize()
{
    // kappa0 on the internal and the processor faces; kappa = 0 on the physical boundary faces
    // (see zeroPhysicalBoundaryKappa).
    NeoN::fill(kappa_.internalVector(), kappa0_);
    {
        const NeoN::localIdx begin = rt_.nfMesh.nBoundaryFaces();
        fillKappa(kappa_, begin, begin + rt_.nfMesh.nProcBoundaryFaces(), kappa0_);
    }
    zeroPhysicalBoundaryKappa(rt_.nfMesh, kappa_);
    logTransition("5 HybridScheme", state_, State::updated);
    state_ = State::updated;
}

void HybridScheme::update(
    const PressureEquation& pressure,
    const KNP& knp,
    const Continuity& /*continuity*/,
    const MomentumOperators& /*momentum*/,
    const EnergyThermo& /*thermo*/
)
{
    const scalar dt = rt_.dt;
    const scalar small = amaxSmall_;

    // amaxSf = max(|ap|, |am|) = max(|u + c|, |u - c|) |Sf| over both sides of the face (KNP
    // characteristic speeds), as in interTwoPhaseCentralFoam::updateKappa and
    // vofTwoPhaseCentralFoam::CharacteristicCourant; ACo is then the characteristic Courant
    // number. (The original pimpleCentralFoam uses max(|phi_o/rho_o|, |phi_n/rho_n|), which has
    // no sound speed at initialization, where phi = linear(rho U) & Sf.) small keeps ACo > 0 in
    // the degenerate case u = c = 0.
    kernels::fieldwise(
        amaxSf_,
        NEON_LAMBDA(const scalar ap, const scalar am) {
            return Kokkos::max(Kokkos::abs(ap), Kokkos::abs(am)) + small;
        },
        knp.ap(),
        knp.am()
    );

    // (16): ACo = deltaCoeffs amaxSf / |Sf| dt.
    kernels::fieldwise(
        ACo_,
        NEON_LAMBDA(const scalar delta, const scalar amax, const scalar magSf) {
            return delta * amax / magSf * dt;
        },
        deltaCoeffs_,
        amaxSf_,
        geometry_.magSf
    );

    // (14): Ma = max(|phi| / (rho_o a_o + rho_n a_n) / |Sf| / (c_o a_o + c_n a_n), 0),
    //       phi = phi_o + phi_n.
    auto weighted = NEON_LAMBDA(const scalar xo, const scalar ao, const scalar xn, const scalar an)
    {
        return xo * ao + xn * an;
    };
    kernels::fieldwise(rhoA_, weighted, pressure.rhoOwn(), knp.alphaOwn(), pressure.rhoNei(), knp.alphaNei());
    kernels::fieldwise(cA_, weighted, knp.cOwn(), knp.alphaOwn(), knp.cNei(), knp.alphaNei());
    kernels::fieldwise(
        Ma_,
        NEON_LAMBDA(const scalar po, const scalar pn) { return Kokkos::abs(po + pn); },
        pressure.phiOwn(),
        pressure.phiNei()
    );
    kernels::fieldwise(
        Ma_,
        NEON_LAMBDA(const scalar magPhi, const scalar rhoA, const scalar magSf, const scalar cA) {
            return Kokkos::max(magPhi / rhoA / magSf / cA, scalar(0));
        },
        Ma_,
        rhoA_,
        geometry_.magSf,
        cA_
    );

    // (13): kappa = min(Ma/ACo, 1) on the internal and the processor faces; kappa = 0 on the
    // physical boundary faces.
    kernels::pointwise(
        kappa_.internalVector(),
        NEON_LAMBDA(const scalar ma, const scalar aco) { return Kokkos::min(ma / aco, scalar(1)); },
        Ma_.internalVector(),
        ACo_.internalVector()
    );
    computeProcKappa(rt_.nfMesh, Ma_, ACo_, kappa_);
    zeroPhysicalBoundaryKappa(rt_.nfMesh, kappa_);

    // The original's kappaFunction can reset kappa to 0 on coupled patches
    // (resetCoupledBoundaries); its updateKappa.H copies only the internal values anyway, so there
    // kappa = 0 on all coupled faces. It is not done here: processor faces have both states, and
    // resetting them makes the solution depend on the decomposition (each rank would convect
    // with its own side's values through the face). A reset is justified for coupled patches
    // whose neighbour state is not a single cell: AMI (area-weighted mix of several cells) and
    // ACMI (partly a physical wall, i.e. one state missing, as on a boundary). NeoN has none of
    // them now; if they appear, reset kappa on those patch types only.
    if (false)
    {
        resetCoupledBoundaries(rt_.nfMesh, kappa_);
    }

    if (printStatistics_)
    {
        // Internal and processor faces: the same faces in serial and parallel runs.
        const auto aco = kernels::minMaxInternalAndProc(ACo_);
        const auto ma = kernels::minMaxInternalAndProc(Ma_);
        const auto kappa = kernels::minMaxInternalAndProc(kappa_);
        NeoN::Logging::info("    max/min FaceAcCourant: {}/{}", aco.max, aco.min);
        NeoN::Logging::info("    max/min Maf: {}/{}", ma.max, ma.min);
        NeoN::Logging::info("    max/min kappa: {}/{}", kappa.max, kappa.min);
    }

    // (12): phi_own = phiP_own + (1 - kappa) phiP_nei, phi_nei = kappa phiP_nei.
    kernels::fieldwise(
        phiOwn_,
        NEON_LAMBDA(const scalar po, const scalar pn, const scalar kappa) {
            return po + (scalar(1) - kappa) * pn;
        },
        pressure.phiOwn(),
        pressure.phiNei(),
        kappa_
    );
    kernels::fieldwise(
        phiNei_,
        NEON_LAMBDA(const scalar pn, const scalar kappa) { return kappa * pn; },
        pressure.phiNei(),
        kappa_
    );
    kernels::fieldwise(
        phi_, NEON_LAMBDA(const scalar a, const scalar b) { return a + b; }, phiOwn_, phiNei_
    );

    logTransition("5 HybridScheme", state_, State::updated);
    state_ = State::updated;
}

void HybridScheme::reset() { state_ = State::outdated; }

} // namespace hcs
