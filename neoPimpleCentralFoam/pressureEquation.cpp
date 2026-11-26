// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#include "blocks.hpp"

namespace hcs
{

namespace
{

NeoN::la::Solver createPressureSolver(const VolScalar& p, const nf::RunTime& rt)
{
    auto dict = NeoFOAM::solverSettings(rt.fvSolutionDict.subDict("solvers"), p.name);
    nf::PDE<NeoN::scalar>::stripNeoFOAMKeys(dict);
    return NeoN::la::Solver(p.exec(), dict);
}

} // namespace

PressureEquation::PressureEquation(
    nf::RunTime& rt, fvcc::VectorCollection& fields, const fvcc::BoundaryContext& ctx
)
    : rt_(rt),
      ctx_(ctx),
      ofP_(
          Foam::IOobject(
              "p",
              rt.mesh.time().timeName(),
              rt.mesh,
              Foam::IOobject::MUST_READ,
              Foam::IOobject::NO_WRITE
          ),
          rt.mesh
      ),
      p_(nf::constructAndRegister(fields, rt, ofP_, false)),
      phiOwn_(makeSurfScalar(rt, "phiP_own")),
      phiNei_(makeSurfScalar(rt, "phiP_nei")),
      pOwn_(makeSurfScalar(rt, "p_own")),
      pNei_(makeSurfScalar(rt, "p_nei")),
      rhoOwn_(makeSurfScalar(rt, "rho_own")),
      rhoNei_(makeSurfScalar(rt, "rho_nei")),
      own_(makeDirection(rt, "own", 1.0)),
      nei_(makeDirection(rt, "nei", -1.0)),
      reconstructP_(interpolationScheme(rt, "reconstruct(p)")),
      lsOwn_(NeoN::la::createEmptyLinearSystem<NeoN::scalar, NeoN::scalar>(rt.nfMesh)),
      lsNei_(NeoN::la::createEmptyLinearSystem<NeoN::scalar, NeoN::scalar>(rt.nfMesh)),
      lsP_(NeoN::la::createEmptyLinearSystem<NeoN::scalar, NeoN::scalar>(rt.nfMesh)),
      pSolver_(createPressureSolver(p_, rt)),
      nNonOrthCorr_(rt.mesh.solutionDict().subDict("PIMPLE").getOrDefault<Foam::label>(
          "nNonOrthogonalCorrectors", 0
      )),
      geometry_(rt),
      rhoU_(makeVolField<NeoN::Vec3>(rt, "rhoU")),
      rhoUf_(makeSurfField<NeoN::Vec3>(rt, "rhoUf")),
      linearV_(rt.exec, rt.nfMesh, NeoN::TokenList({std::string("linear")}))
{
    // The fluxes (19), (20) must include the deferred non-orthogonal correction that the
    // corrected Laplacians put into the right-hand side, as pEqn_own.flux() does.
    lsOwn_.keepFaceFluxCorrection(true);
    lsNei_.keepFaceFluxCorrection(true);
    NeoN::Logging::info(
        "3.2 PressureEquation: read p, nNonOrthogonalCorrectors = {}", nNonOrthCorr_
    );
}

void PressureEquation::assemble(const NeoN::dsl::Expression<NeoN::scalar>& expr, LinearSystem& ls)
{
    auto e = expr;
    e.read(NeoFOAM::expandSchemeDefaults(rt_.fvSchemesDict, e, p_.name));
    ls.reset();
    e.assemble(rt_.t, rt_.dt, ls, rt_.nfMesh);
}

void PressureEquation::assembleParts(const KNP& knp)
{
    auto& phidOwn = readOnlyOperand(knp.phidOwn());
    auto& phidNei = readOnlyOperand(knp.phidNei());
    auto& DpOwn = readOnlyOperand(knp.DpOwn());
    auto& DpNei = readOnlyOperand(knp.DpNei());

    assemble(
        NeoN::dsl::imp::div(phidOwn, p_) - NeoN::dsl::imp::laplacian(DpOwn, p_), lsOwn_
    );
    assemble(
        NeoN::dsl::imp::div(phidNei, p_) - NeoN::dsl::imp::laplacian(DpNei, p_), lsNei_
    );
}

void PressureEquation::initialize(
    const KNP& knp, const Continuity& continuity, const MomentumOperators& momentum
)
{
    // As the original (createCommonCentralFields.H): phi = linearInterpolate(rho*U) & Sf,
    // phi_own = phi, phi_nei = 0. No pressure matrix is involved at initialization.
    kernels::fieldwise(
        rhoU_,
        NEON_LAMBDA(const scalar rho, const NeoN::Vec3& u) { return rho * u; },
        continuity.rho(),
        momentum.U()
    );
    linearV_.interpolate(rhoU_, rhoUf_);
    kernels::fieldwise(
        phiOwn_,
        NEON_LAMBDA(const NeoN::Vec3& f, const NeoN::Vec3& sf) { return f & sf; },
        rhoUf_,
        geometry_.Sf
    );
    kernels::fieldwise(phiNei_, NEON_LAMBDA(const scalar) { return scalar(0); }, phiOwn_);
    reconstructSides(knp);
    logTransition("3.2 PressureEquation", state_, State::solved);
    state_ = State::solved;
}

void PressureEquation::update(const KNP& knp, const EnergyThermo& thermo)
{
    auto& psi = readOnlyOperand(thermo.psi());

    // (7): ddt(psi, p) + [div(phid_own, p) - laplacian(Dp_own, p)]
    //                  + [div(phid_nei, p) - laplacian(Dp_nei, p)] = 0,
    // the total matrix being the sum of its parts, as pEqn = ddt + pEqn_own + pEqn_nei.
    // Each non-orthogonal corrector assembles again with the latest p (the deferred correction
    // of the corrected Laplacians); the fluxes come from the matrices of the last corrector.
    for (int corr = 0; corr <= nNonOrthCorr_; ++corr)
    {
        // The boundary conditions of p see the current phi, U and psi.
        p_.correctBoundaryConditions(ctx_);
        assembleParts(knp);
        assemble(NeoN::dsl::Expression<NeoN::scalar>(NeoN::dsl::imp::ddt(psi, p_)), lsP_);
        kernels::addTo(lsP_, lsOwn_);
        kernels::addTo(lsP_, lsNei_);

        NeoN::fence(p_.exec());
        const auto stats = pSolver_.solve(lsP_, p_.internalVector());
        for (const auto& e : stats.entries)
        {
            NeoN::Logging::info(
                "    p: initial residual = {}, final residual = {}, iterations = {}",
                e.initResNorm,
                e.finalResNorm,
                e.numIter
            );
        }
        p_.correctBoundaryConditions(ctx_);
    }

    // (19), (20) from the same matrices.
    computeFluxes(knp, lsOwn_, lsNei_);

    logTransition("3.2 PressureEquation", state_, State::solved);
    state_ = State::solved;
}

void PressureEquation::computeFluxes(
    const KNP& knp, const LinearSystem& own, const LinearSystem& nei
)
{
    kernels::matrixFlux(own, p_, phiOwn_);
    kernels::matrixFlux(nei, p_, phiNei_);
    reconstructSides(knp);
}

void PressureEquation::reconstructSides(const KNP& knp)
{
    // p reconstructed from both sides, for the velocity corrector (48).
    reconstructP_.interpolate(own_, p_, pOwn_);
    reconstructP_.interpolate(nei_, p_, pNei_);

    // rho_o = p_o psi_o, rho_n = p_n psi_n (used by the hybrid scheme and by delta K).
    auto rhoSide = NEON_LAMBDA(const scalar pS, const scalar psiS) { return pS * psiS; };
    kernels::fieldwise(rhoOwn_, rhoSide, pOwn_, knp.psiOwn());
    kernels::fieldwise(rhoNei_, rhoSide, pNei_, knp.psiNei());
}

void PressureEquation::reset() { state_ = State::outdated; }

void PressureEquation::advanceTime() { fvcc::rotateOldTimes(p_); }

void PressureEquation::write() const { nf::write(p_, rt_.mesh); }

} // namespace hcs
