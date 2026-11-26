// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#include "centralKernels.hpp"

#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/operators.hpp"
#endif

namespace hcs::kernels
{

namespace
{

/// Reduces value over all ranks of a parallel run (a no-op in a serial run).
void reduceAllRanks([[maybe_unused]] scalar& value, [[maybe_unused]] const char* op)
{
#ifdef NF_WITH_MPI_SUPPORT
    NeoN::mpi::Environment env;
    if (!env.isInitialized() || env.sizeRank() < 2) return;
    const std::string o(op);
    const auto reduceOp = o == "max"   ? NeoN::mpi::ReduceOp::Max
                        : o == "min" ? NeoN::mpi::ReduceOp::Min
                                     : NeoN::mpi::ReduceOp::Sum;
    NeoN::mpi::allReduce(value, reduceOp, env.comm());
#endif
}

} // namespace

void faceGeometry(
    const NeoN::UnstructuredMesh& mesh,
    fvcc::SurfaceField<Vec3>& Sf,
    fvcc::SurfaceField<scalar>& magSf
)
{
    const auto exec = Sf.exec();
    {
        auto [sfV, magV] = NeoN::views(Sf.internalVector(), magSf.internalVector());
        const auto [meshSfV, meshMagV] = NeoN::views(mesh.faceNormals(), mesh.faceAreas());
        NeoN::parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) {
                sfV[f] = meshSfV[f];
                magV[f] = meshMagV[f];
            },
            "hcs::faceGeometryInternal"
        );
    }
    {
        auto [sfV, magV] =
            NeoN::views(Sf.boundaryData().value(), magSf.boundaryData().value());
        const auto [meshSfV, meshMagV] =
            NeoN::views(mesh.boundaryMesh().faceNormals(), mesh.boundaryMesh().faceAreas());
        NeoN::parallelFor(
            exec,
            {0, static_cast<localIdx>(sfV.size())},
            NEON_LAMBDA(const localIdx bf) {
                sfV[bf] = meshSfV[bf];
                magV[bf] = meshMagV[bf];
            },
            "hcs::faceGeometryBoundary"
        );
    }
}

void divScalarSf(
    const fvcc::SurfaceField<scalar>& s,
    const fvcc::SurfaceField<Vec3>& Sf,
    fvcc::VolumeField<Vec3>& out
)
{
    const auto& mesh = out.mesh();
    const auto exec = out.exec();
    NeoN::fill(out.internalVector(), NeoN::zero<Vec3>());

    auto outV = out.internalVector().view();
    {
        const auto [sV, sfV, ownV, neiV] =
            NeoN::views(s.internalVector(), Sf.internalVector(), mesh.faceOwners(), mesh.faceNeighbors());
        NeoN::parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) {
                const Vec3 v = sV[f] * sfV[f];
                for (std::size_t c = 0; c < 3; ++c)
                {
                    Kokkos::atomic_add(&outV[ownV[f]][c], v[c]);
                    Kokkos::atomic_sub(&outV[neiV[f]][c], v[c]);
                }
            },
            "hcs::divScalarSfInternal"
        );
    }
    {
        const auto [sV, sfV, ownV] = NeoN::views(
            s.boundaryData().value(), Sf.boundaryData().value(), mesh.boundaryMesh().faceOwners()
        );
        NeoN::parallelFor(
            exec,
            {0, static_cast<localIdx>(sV.size())},
            NEON_LAMBDA(const localIdx bf) {
                const Vec3 v = sV[bf] * sfV[bf];
                for (std::size_t c = 0; c < 3; ++c)
                {
                    Kokkos::atomic_add(&outV[ownV[bf]][c], v[c]);
                }
            },
            "hcs::divScalarSfBoundary"
        );
    }
    {
        const auto volV = mesh.cellVolumes().view();
        NeoN::parallelFor(
            exec,
            {0, mesh.nCells()},
            NEON_LAMBDA(const localIdx c) { outV[c] = outV[c] * (scalar(1) / volV[c]); },
            "hcs::divScalarSfVolume"
        );
    }
}

void velocityGradient(const fvcc::VolumeField<Vec3>& U, fvcc::VolumeField<NeoN::Tensor>& gradU)
{
    NeoN::fill(gradU.internalVector(), NeoN::zero<NeoN::Tensor>());
    NeoN::fill(gradU.boundaryData().value(), NeoN::zero<NeoN::Tensor>());
    fvcc::GaussGreenGrad(U.exec(), U.mesh()).gradTensor(U, gradU);
    // Processor patches: the neighbour cells' gradients (calculated patches keep their values).
    gradU.correctBoundaryConditions();
}

void divTensor(
    const fvcc::VolumeField<NeoN::Tensor>& T,
    const fvcc::SurfaceField<Vec3>& Sf,
    fvcc::VolumeField<Vec3>& out
)
{
    const auto& mesh = out.mesh();
    const auto exec = out.exec();
    const auto geometry = fvcc::GeometryScheme::readOrCreate(mesh);
    NeoN::fill(out.internalVector(), NeoN::zero<Vec3>());
    auto outV = out.internalVector().view();
    {
        const auto [tV, wV, sfV, ownV, neiV] = NeoN::views(
            T.internalVector(),
            geometry->weights().internalVector(),
            Sf.internalVector(),
            mesh.faceOwners(),
            mesh.faceNeighbors()
        );
        NeoN::parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) {
                const auto own = ownV[f];
                const auto nei = neiV[f];
                const NeoN::Tensor tf = wV[f] * tV[own] + (scalar(1) - wV[f]) * tV[nei];
                const Vec3 v = sfV[f] & tf;
                for (std::size_t c = 0; c < 3; ++c)
                {
                    Kokkos::atomic_add(&outV[own][c], v[c]);
                    Kokkos::atomic_sub(&outV[nei][c], v[c]);
                }
            },
            "hcs::divTensorInternal"
        );
    }
    {
        // Physical boundary faces [0, nBoundaryFaces): T_f = the boundary value of T.
        // Processor faces [nBoundaryFaces, nBoundaryFaces + nProcBoundaryFaces): T_f is linearly
        // interpolated between the local cell and the neighbour-rank cell (the boundary value of T
        // after the halo exchange), w being the weight of the local cell, as on an internal face.
        const localIdx nPhysical = mesh.nBoundaryFaces();
        const auto [tB, sfB, ownB, wB, tV] = NeoN::views(
            T.boundaryData().value(),
            Sf.boundaryData().value(),
            mesh.boundaryMesh().faceOwners(),
            geometry->weights().boundaryData().value(),
            T.internalVector()
        );
        NeoN::parallelFor(
            exec,
            {0, static_cast<localIdx>(tB.size())},
            NEON_LAMBDA(const localIdx bf) {
                const auto own = ownB[bf];
                const NeoN::Tensor tf =
                    bf < nPhysical ? tB[bf] : wB[bf] * tV[own] + (scalar(1) - wB[bf]) * tB[bf];
                const Vec3 v = sfB[bf] & tf;
                for (std::size_t c = 0; c < 3; ++c)
                {
                    Kokkos::atomic_add(&outV[own][c], v[c]);
                }
            },
            "hcs::divTensorBoundary"
        );
    }
    const auto volV = mesh.cellVolumes().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) { outV[c] = outV[c] * (scalar(1) / volV[c]); },
        "hcs::divTensorVolume"
    );
}

void surfaceIntegrate(const fvcc::SurfaceField<scalar>& F, fvcc::VolumeField<scalar>& out)
{
    const auto& mesh = out.mesh();
    const auto exec = out.exec();
    NeoN::fill(out.internalVector(), scalar(0));
    auto outV = out.internalVector().view();
    {
        const auto [fV, ownV, neiV] =
            NeoN::views(F.internalVector(), mesh.faceOwners(), mesh.faceNeighbors());
        NeoN::parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) {
                Kokkos::atomic_add(&outV[ownV[f]], fV[f]);
                Kokkos::atomic_sub(&outV[neiV[f]], fV[f]);
            },
            "hcs::surfaceIntegrateInternal"
        );
    }
    {
        const auto [fV, ownV] =
            NeoN::views(F.boundaryData().value(), mesh.boundaryMesh().faceOwners());
        NeoN::parallelFor(
            exec,
            {0, static_cast<localIdx>(fV.size())},
            NEON_LAMBDA(const localIdx bf) { Kokkos::atomic_add(&outV[ownV[bf]], fV[bf]); },
            "hcs::surfaceIntegrateBoundary"
        );
    }
    const auto volV = mesh.cellVolumes().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) { outV[c] /= volV[c]; },
        "hcs::surfaceIntegrateVolume"
    );
}

scalar maxCellCourant(
    const fvcc::SurfaceField<scalar>& phiA,
    const fvcc::SurfaceField<scalar>& phiB,
    const fvcc::VolumeField<scalar>& rho,
    const scalar dt
)
{
    const auto& mesh = rho.mesh();
    const auto exec = rho.exec();
    NeoN::Vector<scalar> sumPhi(exec, mesh.nCells(), scalar(0));
    auto sV = sumPhi.view();
    {
        const auto [aV, bV, ownV, neiV] = NeoN::views(
            phiA.internalVector(), phiB.internalVector(), mesh.faceOwners(), mesh.faceNeighbors()
        );
        NeoN::parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) {
                const scalar m = Kokkos::abs(aV[f] + bV[f]);
                Kokkos::atomic_add(&sV[ownV[f]], m);
                Kokkos::atomic_add(&sV[neiV[f]], m);
            },
            "hcs::cellCourantInternal"
        );
    }
    {
        const auto [aV, bV, ownV] = NeoN::views(
            phiA.boundaryData().value(),
            phiB.boundaryData().value(),
            mesh.boundaryMesh().faceOwners()
        );
        NeoN::parallelFor(
            exec,
            {0, static_cast<localIdx>(aV.size())},
            NEON_LAMBDA(const localIdx bf) {
                Kokkos::atomic_add(&sV[ownV[bf]], Kokkos::abs(aV[bf] + bV[bf]));
            },
            "hcs::cellCourantBoundary"
        );
    }
    pointwise(
        sumPhi,
        NEON_LAMBDA(const scalar s, const scalar r, const scalar v) {
            return scalar(0.5) * dt * s / (r * v);
        },
        sumPhi,
        rho.internalVector(),
        mesh.cellVolumes()
    );
    return minMax(sumPhi).max;
}

void deltaCoeffs(const NeoN::UnstructuredMesh& mesh, fvcc::SurfaceField<scalar>& delta)
{
    const auto exec = delta.exec();
    const auto geometry = fvcc::GeometryScheme::readOrCreate(mesh);
    geometry->ensureFaceDeltas();
    {
        auto dV = delta.internalVector().view();
        const auto [ownD, neiD] = NeoN::views(
            geometry->faceDeltaOwner().internalVector(),
            geometry->faceDeltaNeighbour().internalVector()
        );
        NeoN::parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) {
                const Vec3 d = ownD[f] - neiD[f];
                dV[f] = scalar(1) / Kokkos::sqrt(d & d);
            },
            "hcs::deltaCoeffsInternal"
        );
    }
    {
        auto dV = delta.boundaryData().value().view();
        const auto bD = mesh.boundaryMesh().deltaCoeffs().view();
        NeoN::parallelFor(
            exec,
            {0, static_cast<localIdx>(dV.size())},
            NEON_LAMBDA(const localIdx bf) { dV[bf] = bD[bf]; },
            "hcs::deltaCoeffsBoundary"
        );
    }
}

namespace
{

void addVector(NeoN::Vector<scalar>& dst, const NeoN::Vector<scalar>& src, const char* what)
{
    NF_ASSERT(dst.size() == src.size(), "hcs::addTo: different sizes of " << what);
    auto [dV, sV] = NeoN::views(dst, src);
    NeoN::parallelFor(
        dst.exec(),
        {0, static_cast<localIdx>(dV.size())},
        NEON_LAMBDA(const localIdx i) { dV[i] += sV[i]; },
        "hcs::addTo"
    );
}

} // namespace

void addTo(
    NeoN::la::LinearSystem<scalar, scalar>& dst, const NeoN::la::LinearSystem<scalar, scalar>& src
)
{
    NF_ASSERT(
        dst.matrix().sparsity() == src.matrix().sparsity(),
        "hcs::addTo: the linear systems have different sparsity patterns"
    );
    addVector(dst.matrix().values(), src.matrix().values(), "matrix values");
    addVector(dst.rhs(), src.rhs(), "rhs");
    addVector(dst.boundaryMatrix().values(), src.boundaryMatrix().values(), "boundary matrix");
    addVector(dst.boundaryRhs(), src.boundaryRhs(), "boundary rhs");
    addVector(dst.offDiagonalMatrix().values(), src.offDiagonalMatrix().values(), "coupling");
}

void matrixFlux(
    const NeoN::la::LinearSystem<scalar, scalar>& ls,
    const fvcc::VolumeField<scalar>& psi,
    fvcc::SurfaceField<scalar>& flux
)
{
    const auto& mesh = psi.mesh();
    const auto exec = psi.exec();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();


    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
    const auto values = ls.matrix().values().view();
    const auto [ownV, neiV, psiV] =
        NeoN::views(mesh.faceOwners(), mesh.faceNeighbors(), psi.internalVector());

    const auto& ffcPtr = ls.faceFluxCorrection();
    const bool hasCorrection = ffcPtr != nullptr && ffcPtr->size() == nInternalFaces;
    NeoN::Vector<scalar> noCorrection(exec, 0);
    const auto ffcV = hasCorrection ? ffcPtr->view() : noCorrection.view();

    auto fluxV = flux.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx f) {
            const auto own = ownV[f];
            const auto nei = neiV[f];
            // upperIdx: row own, column nei (OpenFOAM's upper); lowerIdx: row nei, column own.
            const scalar upper = values[ma.upperIdx(own, f)];
            const scalar lower = values[ma.lowerIdx(nei, f)];
            fluxV[f] = upper * psiV[nei] - lower * psiV[own] + (hasCorrection ? ffcV[f] : scalar(0));
        },
        "hcs::matrixFluxInternal"
    );

    const auto [bAV, bRhsV, bOwnV] = NeoN::views(
        ls.boundaryMatrix().values(), ls.boundaryRhs(), mesh.boundaryMesh().faceOwners()
    );
    auto bFluxV = flux.boundaryData().value().view();
    NeoN::parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bf) { bFluxV[bf] = bRhsV[bf] - bAV[bf] * psiV[bOwnV[bf]]; },
        "hcs::matrixFluxBoundary"
    );

    // Processor faces, out of the rank-local cell (OpenFOAM's orientation of a coupled patch):
    // the face's part of the local row, -bA psi_c + coupling psi_ghost, where bA is the diagonal
    // part recorded in the boundary matrix and the coupling coefficient multiplies the
    // neighbour-rank value (the boundary value of psi after the halo exchange).
    const auto nProcFaces = mesh.nProcBoundaryFaces();
    if (nProcFaces > 0)
    {
        const auto couplingV = ls.offDiagonalMatrix().values().view();
        const auto rowOrderV = mesh.boundaryMesh().getRowOrderWriteIndex().view();
        const auto psiGhostV = psi.boundaryData().value().view();
        NeoN::parallelFor(
            exec,
            {0, nProcFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const auto bf = nBoundaryFaces + procFacei;
                bFluxV[bf] = couplingV[rowOrderV[procFacei]] * psiGhostV[bf]
                           - bAV[bf] * psiV[bOwnV[bf]];
            },
            "hcs::matrixFluxProcessor"
        );
    }
}

namespace
{

/// Min/max of f(v[i]) over [begin, end) on this rank only.
template<typename T, typename F>
MinMax localMinMax(const NeoN::Vector<T>& v, const localIdx begin, const localIdx end, F f)
{
    using Result = Kokkos::MinMaxScalar<scalar>;
    Result result;
    // Identity values: the serial executor runs the kernel without initialising the reducer.
    result.min_val = Kokkos::Experimental::finite_max_v<scalar>;
    result.max_val = Kokkos::Experimental::finite_min_v<scalar>;
    if (end <= begin) return MinMax {result.min_val, result.max_val};
    Kokkos::MinMax<scalar> reducer(result);
    const auto vv = v.view();
    NeoN::parallelReduce(
        v.exec(),
        {begin, end},
        NEON_LAMBDA(const localIdx i, Result& r) {
            const scalar x = f(vv[i]);
            if (x < r.min_val) r.min_val = x;
            if (x > r.max_val) r.max_val = x;
        },
        reducer
    );
    return MinMax {result.min_val, result.max_val};
}

MinMax reduceAllRanks(MinMax mm)
{
    reduceAllRanks(mm.min, "min");
    reduceAllRanks(mm.max, "max");
    return mm;
}

template<typename T, typename F>
MinMax reduceMinMax(const NeoN::Vector<T>& v, F f)
{
    return reduceAllRanks(localMinMax(v, 0, static_cast<localIdx>(v.size()), f));
}

} // namespace

scalar domainIntegrate(const NeoN::Vector<scalar>& f, const NeoN::Vector<scalar>& V)
{
    const auto [fV, vV] = NeoN::views(f, V);
    scalar result = 0;
    NeoN::parallelReduce(
        f.exec(),
        {0, static_cast<localIdx>(fV.size())},
        NEON_LAMBDA(const localIdx i, scalar& s) { s += fV[i] * vV[i]; },
        result
    );
    reduceAllRanks(result, "sum");
    return result;
}

scalar domainIntegrateDifference(
    const NeoN::Vector<scalar>& a,
    const NeoN::Vector<scalar>& b,
    const NeoN::Vector<scalar>& V,
    const bool absolute
)
{
    const auto [aV, bV, vV] = NeoN::views(a, b, V);
    scalar result = 0;
    NeoN::parallelReduce(
        a.exec(),
        {0, static_cast<localIdx>(aV.size())},
        NEON_LAMBDA(const localIdx i, scalar& s) {
            const scalar d = aV[i] - bV[i];
            s += (absolute ? Kokkos::abs(d) : d) * vV[i];
        },
        result
    );
    reduceAllRanks(result, "sum");
    return result;
}

void copyRange(NeoN::Vector<Vec3>& dst, const NeoN::Vector<Vec3>& src, localIdx begin, localIdx end)
{
    auto dV = dst.view();
    const auto sV = src.view();
    NeoN::parallelFor(
        dst.exec(),
        {begin, end},
        NEON_LAMBDA(const localIdx i) { dV[i] = sV[i]; },
        "hcs::copyRange"
    );
}

void copyAt(
    NeoN::Vector<Vec3>& dst, const NeoN::Vector<Vec3>& src, const NeoN::Vector<localIdx>& cells
)
{
    auto dV = dst.view();
    const auto [sV, cV] = NeoN::views(src, cells);
    NeoN::parallelFor(
        dst.exec(),
        {0, static_cast<localIdx>(cV.size())},
        NEON_LAMBDA(const localIdx i) { dV[cV[i]] = sV[cV[i]]; },
        "hcs::copyAt"
    );
}

MinMax minMax(const NeoN::Vector<scalar>& v)
{
    return reduceMinMax(v, NEON_LAMBDA(const scalar x) { return x; });
}

MinMax minMaxMag(const NeoN::Vector<Vec3>& v)
{
    return reduceMinMax(v, NEON_LAMBDA(const Vec3& x) { return Kokkos::sqrt(x & x); });
}

void characteristicCourant(
    const fvcc::SurfaceField<scalar>& ap,
    const fvcc::SurfaceField<scalar>& am,
    const fvcc::SurfaceField<scalar>& deltaCoeffs,
    const fvcc::SurfaceField<scalar>& magSf,
    const scalar dt,
    fvcc::SurfaceField<scalar>& out
)
{
    fieldwise(
        out,
        NEON_LAMBDA(const scalar a, const scalar b, const scalar delta, const scalar mSf) {
            return Kokkos::max(Kokkos::abs(a), Kokkos::abs(b)) * delta / mSf * dt;
        },
        ap,
        am,
        deltaCoeffs,
        magSf
    );
}

MinMax minMaxInternalAndProc(const fvcc::SurfaceField<scalar>& s)
{
    const auto& mesh = s.mesh();
    auto identity = NEON_LAMBDA(const scalar x) { return x; };
    const auto& internal = s.internalVector();
    const MinMax a = localMinMax(internal, 0, static_cast<localIdx>(internal.size()), identity);
    const localIdx begin = mesh.nBoundaryFaces();
    const localIdx end = begin + mesh.nProcBoundaryFaces();
    const MinMax b = localMinMax(s.boundaryData().value(), begin, end, identity);
    return reduceAllRanks(MinMax {Kokkos::min(a.min, b.min), Kokkos::max(a.max, b.max)});
}

} // namespace hcs::kernels
