// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

/*---------------------------------------------------------------------------*\
    Device kernels of the relations. Depends on NeoN only.

    fieldwise(out, f, a, ...) evaluates out = f(a, ...) value by value on the
    internal values and on the boundary values of fields of the same kind
    (volume or surface). The boundary values of processor patches hold the
    neighbour values, so a pointwise relation stays valid there as well.
\*---------------------------------------------------------------------------*/

#pragma once

#include "NeoN/NeoN.hpp"

namespace hcs::kernels
{

namespace fvcc = NeoN::finiteVolume::cellCentred;
using NeoN::localIdx;
using NeoN::scalar;
using NeoN::Vec3;

// * * * * * * * * * * * * * * * * pointwise * * * * * * * * * * * * * * * * //

template<typename O, typename A, typename F>
void pointwise(NeoN::Vector<O>& out, F f, const NeoN::Vector<A>& a)
{
    auto [ov, av] = NeoN::views(out, a);
    NeoN::parallelFor(
        out.exec(),
        {0, static_cast<localIdx>(ov.size())},
        NEON_LAMBDA(const localIdx i) { ov[i] = f(av[i]); },
        "hcs::pointwise1"
    );
}

template<typename O, typename A, typename B, typename F>
void pointwise(NeoN::Vector<O>& out, F f, const NeoN::Vector<A>& a, const NeoN::Vector<B>& b)
{
    auto [ov, av, bv] = NeoN::views(out, a, b);
    NeoN::parallelFor(
        out.exec(),
        {0, static_cast<localIdx>(ov.size())},
        NEON_LAMBDA(const localIdx i) { ov[i] = f(av[i], bv[i]); },
        "hcs::pointwise2"
    );
}

template<typename O, typename A, typename B, typename C, typename F>
void pointwise(
    NeoN::Vector<O>& out,
    F f,
    const NeoN::Vector<A>& a,
    const NeoN::Vector<B>& b,
    const NeoN::Vector<C>& c
)
{
    auto [ov, av, bv, cv] = NeoN::views(out, a, b, c);
    NeoN::parallelFor(
        out.exec(),
        {0, static_cast<localIdx>(ov.size())},
        NEON_LAMBDA(const localIdx i) { ov[i] = f(av[i], bv[i], cv[i]); },
        "hcs::pointwise3"
    );
}

template<typename O, typename A, typename B, typename C, typename D, typename F>
void pointwise(
    NeoN::Vector<O>& out,
    F f,
    const NeoN::Vector<A>& a,
    const NeoN::Vector<B>& b,
    const NeoN::Vector<C>& c,
    const NeoN::Vector<D>& d
)
{
    auto [ov, av, bv, cv, dv] = NeoN::views(out, a, b, c, d);
    NeoN::parallelFor(
        out.exec(),
        {0, static_cast<localIdx>(ov.size())},
        NEON_LAMBDA(const localIdx i) { ov[i] = f(av[i], bv[i], cv[i], dv[i]); },
        "hcs::pointwise4"
    );
}

/// out = f(in...) on the internal and on the boundary values.
template<typename OutField, typename F, typename... InFields>
void fieldwise(OutField& out, F f, const InFields&... in)
{
    pointwise(out.internalVector(), f, in.internalVector()...);
    pointwise(out.boundaryData().value(), f, in.boundaryData().value()...);
}

// * * * * * * * * * * * * * * * * geometry * * * * * * * * * * * * * * * * //

/// Sf and |Sf| as surface fields (internal faces from the mesh, boundary and processor faces
/// from the boundary mesh).
void faceGeometry(
    const NeoN::UnstructuredMesh& mesh,
    fvcc::SurfaceField<Vec3>& Sf,
    fvcc::SurfaceField<scalar>& magSf
);

// * * * * * * * * * * * * * * * * operators * * * * * * * * * * * * * * * //

/// out = fvc::div(s*Sf) = (1/V) sum_f (+/-) s_f Sf_f.
void divScalarSf(
    const fvcc::SurfaceField<scalar>& s,
    const fvcc::SurfaceField<Vec3>& Sf,
    fvcc::VolumeField<Vec3>& out
);

/// gradU(i,j) = dU_i/dx_j by Gauss linear, cells and boundary faces (the transpose of
/// OpenFOAM's fvc::grad(U)).
void velocityGradient(const fvcc::VolumeField<Vec3>& U, fvcc::VolumeField<NeoN::Tensor>& gradU);

/// out = fvc::div(T) = (1/V) sum_f (+/-) Sf . T_f, T_f linearly interpolated with the mesh weights
/// on the internal and the processor faces (the neighbour-rank value being the boundary value of
/// T after the halo exchange); the boundary values of T on the physical boundary faces.
void divTensor(
    const fvcc::VolumeField<NeoN::Tensor>& T,
    const fvcc::SurfaceField<Vec3>& Sf,
    fvcc::VolumeField<Vec3>& out
);

/// out = fvc::surfaceIntegrate(F) = (1/V) sum_f (+/-) F_f.
void surfaceIntegrate(const fvcc::SurfaceField<scalar>& F, fvcc::VolumeField<scalar>& out);

/// Cell Courant number of OpenFOAM's compressible solvers:
/// max_c 0.5 dt sum_f |phiA_f + phiB_f| / (rho_c V_c), the mass flux split into two parts.
scalar maxCellCourant(
    const fvcc::SurfaceField<scalar>& phiA,
    const fvcc::SurfaceField<scalar>& phiB,
    const fvcc::VolumeField<scalar>& rho,
    scalar dt
);

/// 1/|C_nei - C_own| on the internal faces, the boundary-mesh delta coefficients on the
/// boundary faces (OpenFOAM's surfaceInterpolation::deltaCoeffs()).
void deltaCoeffs(const NeoN::UnstructuredMesh& mesh, fvcc::SurfaceField<scalar>& delta);

/// dst += src for two linear systems of the same mesh (same sparsity): matrix values, rhs,
/// boundary matrix and boundary rhs, processor coupling coefficients.
void addTo(
    NeoN::la::LinearSystem<scalar, scalar>& dst, const NeoN::la::LinearSystem<scalar, scalar>& src
);

/**
 * OpenFOAM's fvMatrix::flux() of an assembled scalar equation for the field psi:
 * internal faces A(own,nei) psi_nei - A(nei,own) psi_own (+ the deferred non-orthogonal
 * correction, if the Laplacian kept it), boundary faces bRhs - bA psi_own, processor faces
 * coupling psi_ghost - bA psi_own (out of the rank-local cell). The deferred non-orthogonal
 * correction is not applied on processor faces (NeoN keeps it for internal faces only).
 */
void matrixFlux(
    const NeoN::la::LinearSystem<scalar, scalar>& ls,
    const fvcc::VolumeField<scalar>& psi,
    fvcc::SurfaceField<scalar>& flux
);

// * * * * * * * * * * * * * * * * reductions * * * * * * * * * * * * * * * //

/// sum_i f_i V_i (OpenFOAM's fvc::domainIntegrate, without the parallel reduction).
scalar domainIntegrate(const NeoN::Vector<scalar>& f, const NeoN::Vector<scalar>& V);

/// sum_i (a_i - b_i) V_i, or sum_i |a_i - b_i| V_i with absolute = true.
scalar domainIntegrateDifference(
    const NeoN::Vector<scalar>& a,
    const NeoN::Vector<scalar>& b,
    const NeoN::Vector<scalar>& V,
    bool absolute
);

/// dst_i = src_i for i in [begin, end).
void copyRange(NeoN::Vector<Vec3>& dst, const NeoN::Vector<Vec3>& src, localIdx begin, localIdx end);

/// dst_c = src_c for the cells c listed in cells.
void copyAt(
    NeoN::Vector<Vec3>& dst, const NeoN::Vector<Vec3>& src, const NeoN::Vector<localIdx>& cells
);

struct MinMax
{
    scalar min;
    scalar max;
};

/// The reductions below are global: in a parallel run they are reduced over all ranks.

/// Minimum and maximum of the values.
MinMax minMax(const NeoN::Vector<scalar>& v);

/// Minimum and maximum of the magnitudes of the values.
MinMax minMaxMag(const NeoN::Vector<Vec3>& v);

/// out = max(|ap|, |am|) deltaCoeffs/|Sf| dt, i.e. max(|u + c|, |u - c|) dt/dx with the KNP
/// characteristic speeds ap, am (both sides of the face): the characteristic Courant number.
void characteristicCourant(
    const fvcc::SurfaceField<scalar>& ap,
    const fvcc::SurfaceField<scalar>& am,
    const fvcc::SurfaceField<scalar>& deltaCoeffs,
    const fvcc::SurfaceField<scalar>& magSf,
    scalar dt,
    fvcc::SurfaceField<scalar>& out
);

/// Min/max of a face field over the internal and the processor faces (not the physical boundary
/// faces), so that serial and parallel runs report the same values.
MinMax minMaxInternalAndProc(const fvcc::SurfaceField<scalar>& s);

} // namespace hcs::kernels
