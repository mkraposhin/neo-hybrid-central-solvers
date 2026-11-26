// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

/*---------------------------------------------------------------------------*\
    The blocks of the neoPimpleCentralFoam queue.

    Each block owns the fields its relations compute (fields with a time
    derivative are stored in NeoN's field database, the block keeps the only
    non-const reference), receives the blocks it reads as const references
    in its transitions, and changes behaviour only according to its own state
    or the state of the blocks passed to it.

    Queue of one time step:
        1 TimeStep, 2 Predictor (2.1 Continuity, 2.2 MomentumOperators,
        2.3 EnergyThermo), 3 PVC (3.1 KNP, 3.2 PressureEquation),
        4 Corrector (4.1 = 2.2, 4.2 = 2.1), 5 HybridScheme,
        6 MechanicalSources, 7 EndOfIteration.
\*---------------------------------------------------------------------------*/

#pragma once

#include "hcsTypes.hpp"
#include "perfectGas.hpp"

#include "NeoFOAM/datastructures/solver.hpp"

#include "fvCFD.H"

namespace hcs
{

/// controlDict switch 'printStatistics' (default on): max Co after the time step selection and
/// max/min of p, T, mag(U) at the end of each iteration.
inline bool printStatistics(const nf::RunTime& rt)
{
    return rt.mesh.time().controlDict().getOrDefault<Foam::Switch>("printStatistics", true);
}

class Continuity;
class MomentumOperators;
class EnergyThermo;
class KNP;
class PressureEquation;
class HybridScheme;
class MechanicalSources;

// * * * * * * * * * * * * * * * * 3.2 * * * * * * * * * * * * * * * * * * //

/// Pressure balance (7) and mass fluxes phi_o, phi_n (19), (20). Owns p.
class PressureEquation
{
public:

    enum class State
    {
        initial,
        solved,
        outdated
    };

    friend const char* toString(State s)
    {
        switch (s)
        {
        case State::initial:
            return "initial";
        case State::solved:
            return "solved";
        default:
            return "outdated";
        }
    }

    /// ctx: the fields the boundary conditions of p may need (phi, U, psi, gamma).
    PressureEquation(
        nf::RunTime& rt, fvcc::VectorCollection& fields, const fvcc::BoundaryContext& ctx
    );

    /// Initialization step 6, as the original (createCommonCentralFields.H):
    /// phi_o = linear(rho U) & Sf, phi_n = 0; p_o, p_n, rho_o, rho_n from the p read at construction.
    void initialize(const KNP& knp, const Continuity& continuity, const MomentumOperators& momentum);

    /// (7), then (19), (20).
    void update(const KNP& knp, const EnergyThermo& thermo);

    void reset();

    State state() const { return state_; }
    const VolScalar& p() const { return p_; }
    const SurfScalar& phiOwn() const { return phiOwn_; }
    const SurfScalar& phiNei() const { return phiNei_; }
    const SurfScalar& pOwn() const { return pOwn_; }
    const SurfScalar& pNei() const { return pNei_; }
    const SurfScalar& rhoOwn() const { return rhoOwn_; }
    const SurfScalar& rhoNei() const { return rhoNei_; }

    /// Moves the current p to the old-time level (called by TimeStep only).
    void advanceTime();

    void write() const;

private:

    nf::RunTime& rt_;
    State state_ = State::initial;
    const fvcc::BoundaryContext& ctx_;
    Foam::volScalarField ofP_; ///< output mirror; keeps the BC types for writing
    VolScalar& p_;             ///< stored in the field database (old time needed)
    SurfScalar phiOwn_;        ///< (19)
    SurfScalar phiNei_;        ///< (20)
    SurfScalar pOwn_;          ///< p reconstructed from the owner side (input of (48))
    SurfScalar pNei_;          ///< p reconstructed from the neighbour side (input of (48))
    SurfScalar rhoOwn_;        ///< p_o psi_o
    SurfScalar rhoNei_;        ///< p_n psi_n
    SurfScalar own_;
    SurfScalar nei_;
    fvcc::SurfaceInterpolation<NeoN::scalar> reconstructP_; ///< scheme 'reconstruct(p)'

    using LinearSystem = NeoN::la::LinearSystem<NeoN::scalar, NeoN::scalar>;
    LinearSystem lsOwn_;       ///< div(phid_own, p) - laplacian(Dp_own, p)
    LinearSystem lsNei_;       ///< div(phid_nei, p) - laplacian(Dp_nei, p)
    LinearSystem lsP_;         ///< (7) = ddt(psi, p) + lsOwn_ + lsNei_
    NeoN::la::Solver pSolver_; ///< fvSolution/solvers/p
    int nNonOrthCorr_;         ///< PIMPLE.nNonOrthogonalCorrectors (default 0)
    FaceGeometry geometry_;    ///< Sf for the initial phi
    VolVector rhoU_;           ///< rho U, for the initial phi
    SurfVector rhoUf_;         ///< linear(rho U)
    fvcc::SurfaceInterpolation<NeoN::Vec3> linearV_; ///< 'linear', for the initial phi

    /// p_o, p_n (scheme 'reconstruct(p)') and rho_o = p_o psi_o, rho_n = p_n psi_n.
    void reconstructSides(const KNP& knp);

    /// Assembles the owner and neighbour parts of (7) with the current p.
    void assembleParts(const KNP& knp);

    /// expr into ls (reset first), with the schemes of fvSchemes.
    void assemble(const NeoN::dsl::Expression<NeoN::scalar>& expr, LinearSystem& ls);

    /// (19), (20) from the matrices that gave p: pEqn_own.flux(), pEqn_nei.flux();
    /// p_o, p_n, rho_o, rho_n.
    void computeFluxes(const KNP& knp, const LinearSystem& own, const LinearSystem& nei);
};

// * * * * * * * * * * * * * * * * 3.1 * * * * * * * * * * * * * * * * * * //

/// KNP weights and compressibility coefficients: (21)-(24), (27), (28), (30)-(38), (47).
class KNP
{
public:

    enum class State
    {
        initial,
        updated,
        outdated
    };

    friend const char* toString(State s)
    {
        switch (s)
        {
        case State::initial:
            return "initial";
        case State::updated:
            return "updated";
        default:
            return "outdated";
        }
    }

    /// Kurganov's flux scheme (the only one).
    explicit KNP(nf::RunTime& rt);

    void initialize(
        const Continuity& continuity,
        const MomentumOperators& momentum,
        const EnergyThermo& thermo,
        const HybridScheme& hybrid
    );

    void update(
        const Continuity& continuity,
        const MomentumOperators& momentum,
        const EnergyThermo& thermo,
        const HybridScheme& hybrid
    );

    void reset();

    State state() const { return state_; }
    const SurfScalar& alphaOwn() const { return alphaOwn_; }
    const SurfScalar& alphaNei() const { return alphaNei_; }
    const SurfScalar& cOwn() const { return cOwn_; }
    const SurfScalar& cNei() const { return cNei_; }
    const SurfScalar& phidOwn() const { return phidOwn_; }
    const SurfScalar& phidNei() const { return phidNei_; }
    const SurfScalar& DpOwn() const { return DpOwn_; }
    const SurfScalar& DpNei() const { return DpNei_; }
    const SurfScalar& psiOwn() const { return psiOwn_; }
    const SurfScalar& psiNei() const { return psiNei_; }
    const SurfScalar& ap() const { return ap_; }   ///< (36)
    const SurfScalar& am() const { return am_; }   ///< (35)

private:

    nf::RunTime& rt_;
    State state_ = State::initial;
    FaceGeometry geometry_;
    SurfScalar cOwn_;    ///< (37)
    SurfScalar cNei_;    ///< (38)
    VolScalar W_;        ///< (30)
    SurfScalar alphaOwn_; ///< (34)
    SurfScalar alphaNei_; ///< (47)
    SurfScalar aSf_;     ///< (33)
    SurfScalar psiOwn_;  ///< (27)
    SurfScalar psiNei_;  ///< (28)
    SurfScalar phidOwn_; ///< (21)
    SurfScalar phidNei_; ///< (22)
    SurfScalar DpOwn_;   ///< (23)
    SurfScalar DpNei_;   ///< (24)

    // Intermediate values of the relations above.
    SurfScalar wOwn_;    ///< W reconstructed from the owner side
    SurfScalar wNei_;    ///< W reconstructed from the neighbour side
    SurfScalar phivOwn_; ///< (31), (32): volumetric fluxes from U, then from H/A
    SurfScalar phivNei_;
    SurfScalar cSfOwn_;  ///< c |Sf|
    SurfScalar cSfNei_;
    SurfScalar ap_;      ///< (36)
    SurfScalar am_;      ///< (35)
    SurfScalar psiLin_;  ///< psi by central differencing
    VolVector WU_;       ///< W U, then W H/A
    SurfVector WUOwn_;
    SurfVector WUNei_;
    VolScalar WrAU_;     ///< W/A
    SurfScalar own_;
    SurfScalar nei_;
    fvcc::SurfaceInterpolation<NeoN::scalar> reconstructW_;   ///< 'reconstruct(W)'
    fvcc::SurfaceInterpolation<NeoN::Vec3> reconstructU_;     ///< 'reconstruct(U)'
    fvcc::SurfaceInterpolation<NeoN::scalar> reconstructPsi_; ///< 'reconstruct(psi)'
    fvcc::SurfaceInterpolation<NeoN::scalar> reconstructDp_;  ///< 'reconstruct(Dp)'
    fvcc::SurfaceInterpolation<NeoN::scalar> linear_;

    /// W reconstructed from both sides weights the face velocity: ((W U)_side / W_side) . Sf.
    void volumetricFluxes(const VolVector& velocity);

    void compute(
        const Continuity& continuity,
        const MomentumOperators& momentum,
        const EnergyThermo& thermo,
        const HybridScheme& hybrid
    );
};

// * * * * * * * * * * * * * * * * * 3 * * * * * * * * * * * * * * * * * * //

/// Pressure-velocity coupling: composite of 3.1 KNP and 3.2 PressureEquation.
class PVC
{
public:

    PVC(nf::RunTime& rt, fvcc::VectorCollection& fields, const fvcc::BoundaryContext& ctx);

    void update(
        const Continuity& continuity,
        const MomentumOperators& momentum,
        const EnergyThermo& thermo,
        const HybridScheme& hybrid
    );

    void reset();

    KNP& knp() { return knp_; }
    const KNP& knp() const { return knp_; }
    PressureEquation& pressure() { return pressure_; }
    const PressureEquation& pressure() const { return pressure_; }

private:

    KNP knp_;
    PressureEquation pressure_;
};

// * * * * * * * * * * * * * * * * 2.1 * * * * * * * * * * * * * * * * * * //

/// Continuity (1) and density corrector (11), one object (2.1 = 4.2). Owns rho.
class Continuity
{
public:

    enum class State
    {
        initial,
        transported, ///< after (1)
        closed,      ///< after (11)
        outdated
    };

    friend const char* toString(State s)
    {
        switch (s)
        {
        case State::initial:
            return "initial";
        case State::transported:
            return "transported";
        case State::closed:
            return "closed";
        default:
            return "outdated";
        }
    }

    Continuity(nf::RunTime& rt, fvcc::VectorCollection& fields);

    /// Initialization step 2: rho via (11).
    void initialize(const EnergyThermo& thermo, const PressureEquation& pressure);

    /// (1) while the pressure equation is not solved in this iteration, (11) afterwards.
    void update(
        const EnergyThermo& thermo,
        const PressureEquation& pressure,
        const HybridScheme& hybrid
    );

    void reset();

    State state() const { return state_; }
    const VolScalar& rho() const { return rho_; }

    /// Adds the global continuity error of the last closure to the cumulative one and
    /// returns it (called once per time step, after the last outer iteration).
    scalar accumulateContinuityError();

    void advanceTime();

    void write() const;

private:

    nf::RunTime& rt_;
    State state_ = State::initial;
    bool printStatistics_;
    Foam::volScalarField ofRho_; ///< output mirror
    VolScalar& rho_;             ///< stored in the field database (old time needed)
    SurfScalar phi_;             ///< scratch: phi_own + phi_nei
    VolScalar rhoT_;             ///< scratch: psi p, for the continuity errors
    nf::Solver<NeoN::scalar> solver_;
    scalar globalContErr_ = 0;
    scalar cumulativeContErr_ = 0;

    /// (1) with phi = phiA + phiB.
    void transport(const SurfScalar& phiA, const SurfScalar& phiB);

    /// (11); with diagnose, first the continuity errors of OpenFOAM's centralContinuityErrs.H:
    /// rho transported with the mass fluxes of the pressure equation against psi p.
    void close(const EnergyThermo& thermo, const PressureEquation& pressure, bool diagnose);
};

// * * * * * * * * * * * * * * * * 2.2 * * * * * * * * * * * * * * * * * * //

/// Momentum operators (2), (25), (26) and velocity corrector (48), one object (2.2 = 4.1).
/// Owns U.
class MomentumOperators
{
public:

    enum class State
    {
        initial,
        assembled, ///< after (2), (25), (26)
        corrected, ///< after (48)
        outdated
    };

    friend const char* toString(State s)
    {
        switch (s)
        {
        case State::initial:
            return "initial";
        case State::assembled:
            return "assembled";
        case State::corrected:
            return "corrected";
        default:
            return "outdated";
        }
    }

    MomentumOperators(
        nf::RunTime& rt,
        fvcc::VectorCollection& fields,
        const PerfectGas& gas,
        const fvcc::BoundaryContext& ctx
    );

    /// Initialization step 3: 1/A = dt/(rho V), H/A = U.
    void initialize(const Continuity& continuity);

    /// Assembles while the pressure equation is not solved in this iteration, corrects after.
    void update(
        const Continuity& continuity,
        const PressureEquation& pressure,
        const KNP& knp,
        const HybridScheme& hybrid
    );

    void reset();

    State state() const { return state_; }
    const PerfectGas& gas() const { return gas_; }
    const VolVector& U() const { return U_; }
    const VolScalar& rAU() const { return rAU_; }
    const VolVector& HbyA() const { return HbyA_; }

    void advanceTime();

    void write() const;

private:

    nf::RunTime& rt_;
    PerfectGas gas_;
    const fvcc::BoundaryContext& ctx_;
    State state_ = State::initial;
    Foam::volVectorField ofU_; ///< read at construction; output mirror
    VolVector& U_;             ///< stored in the field database (old time needed)
    VolScalar rAU_;            ///< 1/A (25)
    VolVector HbyA_;           ///< H/A (26), boundary conditions of U
    FaceGeometry geometry_;
    SurfScalar pf_;            ///< alpha_o p_o + alpha_n p_n
    VolVector gradp_;          ///< div(pf Sf)
    nf::Solver<NeoN::Vec3> solver_; ///< assembles (2); the balance is never solved
    // Viscous term of (2), divDevRhoReff(U) = -laplacian(mu, U) - div(mu dev2(T(grad(U)))).
    SurfScalar muEff_;               ///< mu on the faces
    fvcc::VolumeField<NeoN::Tensor> gradU_;
    fvcc::VolumeField<NeoN::Tensor> stress_; ///< mu dev2(T(grad(U)))
    VolVector viscSource_;           ///< -div(mu dev2(T(grad(U))))
    // Pressure-gradient limiter of (48): in cells of bad quality, grad(p) replaces
    // div((alpha_o p_o + alpha_n p_n) Sf).
    NeoN::Vector<NeoN::localIdx> badQualityCells_;
    VolVector limGradP_;             ///< Gauss linear grad(p)

    /// Cells with nonOrthogonality >= PIMPLE.nonOrthogonalityThreshold or
    /// skewness >= PIMPLE.skewnessThreshold (OpenFOAM's cellQuality).
    static NeoN::Vector<NeoN::localIdx> markBadQualityCells(const nf::RunTime& rt);

    void assemble(const Continuity& continuity, const PressureEquation& pressure, const HybridScheme& hybrid);
    void correct(const PressureEquation& pressure, const KNP& knp);
};

// * * * * * * * * * * * * * * * * 2.3 * * * * * * * * * * * * * * * * * * //

/// Energy balance (3) and thermodynamics (8)-(10), (15), (29), (39)-(46). Owns h, T, psi, c.
class EnergyThermo
{
public:

    enum class State
    {
        initial,
        updated,
        outdated
    };

    friend const char* toString(State s)
    {
        switch (s)
        {
        case State::initial:
            return "initial";
        case State::updated:
            return "updated";
        default:
            return "outdated";
        }
    }

    EnergyThermo(
        nf::RunTime& rt,
        fvcc::VectorCollection& fields,
        const PerfectGas& gas,
        const fvcc::BoundaryContext& ctx
    );

    /// Initialization step 1: constants, psi, c, T_o, T_n (h = h(T) is set at construction,
    /// because the boundary values of h follow from those of T).
    void initialize();

    void update(
        const Continuity& continuity,
        const MomentumOperators& momentum,
        const HybridScheme& hybrid,
        const MechanicalSources& sources
    );

    void reset();

    State state() const { return state_; }
    const PerfectGas& gas() const { return gas_; }
    const VolScalar& T() const { return T_; }
    const VolScalar& h() const { return h_; }
    const VolScalar& psi() const { return psi_; }
    const VolScalar& gammaField() const { return gammaField_; }
    const VolScalar& c() const { return c_; }
    const SurfScalar& TOwn() const { return TOwn_; }
    const SurfScalar& TNei() const { return TNei_; }

    void advanceTime();

    void write() const;

private:

    nf::RunTime& rt_;
    PerfectGas gas_;
    const fvcc::BoundaryContext& ctx_;
    State state_ = State::initial;
    Foam::volScalarField ofT_; ///< read at construction; output mirror
    VolScalar T_;              ///< (10)
    VolScalar& h_;             ///< stored in the field database (old time needed)
    VolScalar& psi_;           ///< (29), stored in the field database (old time needed by (7))
    VolScalar c_;              ///< (15)
    scalar Cv_ = 0;            ///< (8); constant, so the face values (41)-(46) equal the cell ones
    scalar gamma_ = 0;         ///< (9)
    VolScalar gammaField_;     ///< gamma as a field, for the boundary conditions
    SurfScalar TOwn_;          ///< (39)
    SurfScalar TNei_;          ///< (40)
    SurfScalar alphaEff_;      ///< mu/Pr on the faces (constant)
    SurfScalar own_;           ///< +1: owner-side reconstruction
    SurfScalar nei_;           ///< -1: neighbour-side reconstruction
    fvcc::SurfaceInterpolation<NeoN::scalar> reconstructT_; ///< scheme 'reconstruct(psi)'
    nf::Solver<NeoN::scalar> solver_;

    /// Registers h with the boundary conditions derived from those of T and the values h = Cp T.
    static VolScalar&
    createEnthalpy(nf::RunTime& rt, fvcc::VectorCollection& fields, const Foam::volScalarField& ofT, const PerfectGas& gas);

    /// (10), then (29), (15), (39), (40) from the current h (or T at initialization).
    void computeState(bool fromEnthalpy);
};

// * * * * * * * * * * * * * * * * * 2 * * * * * * * * * * * * * * * * * * //

/// Predictor: composite of 2.1 Continuity, 2.2 MomentumOperators, 2.3 EnergyThermo.
class Predictor
{
public:

    Predictor(
        nf::RunTime& rt,
        fvcc::VectorCollection& fields,
        const PerfectGas& gas,
        const fvcc::BoundaryContext& ctx
    );

    /// 2.1 runs only in the first outer iteration of a time step.
    void update(
        bool firstIteration,
        const PressureEquation& pressure,
        const KNP& knp,
        const HybridScheme& hybrid,
        const MechanicalSources& sources
    );

    void reset();

    Continuity& continuity() { return continuity_; }
    const Continuity& continuity() const { return continuity_; }
    MomentumOperators& momentum() { return momentum_; }
    const MomentumOperators& momentum() const { return momentum_; }
    EnergyThermo& thermo() { return thermo_; }
    const EnergyThermo& thermo() const { return thermo_; }

private:

    Continuity continuity_;
    MomentumOperators momentum_;
    EnergyThermo thermo_;
};

// * * * * * * * * * * * * * * * * * 5 * * * * * * * * * * * * * * * * * * //

/// Hybrid scheme: Ma (14), ACo (16), kappa (13), blended fluxes phi (12). Owns kappa, phi.
class HybridScheme
{
public:

    enum class State
    {
        initial,
        updated,
        outdated
    };

    friend const char* toString(State s)
    {
        switch (s)
        {
        case State::initial:
            return "initial";
        case State::updated:
            return "updated";
        default:
            return "outdated";
        }
    }

    /// Reads PIMPLE.kappa0 (default 0, within [0, 1]).
    explicit HybridScheme(nf::RunTime& rt);

    /// Initialization step 4: kappa = kappa0.
    void initialize();

    void update(
        const PressureEquation& pressure,
        const KNP& knp,
        const Continuity& continuity,
        const MomentumOperators& momentum,
        const EnergyThermo& thermo
    );

    void reset();

    State state() const { return state_; }
    const SurfScalar& kappa() const { return kappa_; }
    const SurfScalar& phiOwn() const { return phiOwn_; }
    const SurfScalar& phiNei() const { return phiNei_; }
    const SurfScalar& phi() const { return phi_; }

private:

    nf::RunTime& rt_;
    State state_ = State::initial;
    scalar kappa0_;
    bool printStatistics_;
    FaceGeometry geometry_;
    SurfScalar deltaCoeffs_; ///< 1/|d|
    scalar amaxSmall_;       ///< SMALL min|Sf|, keeps ACo positive
    SurfScalar kappa_;  ///< (13)
    SurfScalar phiOwn_; ///< (12), owner part of the blended flux
    SurfScalar phiNei_; ///< (12), neighbour part of the blended flux
    SurfScalar phi_;    ///< phi_own + phi_nei, the mass flux seen by the boundary conditions
    SurfScalar amaxSf_; ///< max(|ap|, |am|) + amaxSmall (characteristic speed times |Sf|)
    SurfScalar ACo_;    ///< (16), face acoustic Courant number
    SurfScalar Ma_;     ///< (14), face Mach number
    SurfScalar rhoA_;   ///< rho_o a_o + rho_n a_n
    SurfScalar cA_;     ///< c_o a_o + c_n a_n
};

// * * * * * * * * * * * * * * * * * 6 * * * * * * * * * * * * * * * * * * //

/// Mechanical sources: K (5), delta K (4), delta p (6). Owns K.
class MechanicalSources
{
public:

    enum class State
    {
        initial,
        updated,
        outdated
    };

    friend const char* toString(State s)
    {
        switch (s)
        {
        case State::initial:
            return "initial";
        case State::updated:
            return "updated";
        default:
            return "outdated";
        }
    }

    MechanicalSources(nf::RunTime& rt, fvcc::VectorCollection& fields);

    /// Initialization step 8: K = U.U/2, delta K = delta p = 0.
    void initialize(const MomentumOperators& momentum);

    void update(
        const MomentumOperators& momentum,
        const Continuity& continuity,
        const PressureEquation& pressure,
        const HybridScheme& hybrid
    );

    void reset();

    State state() const { return state_; }
    const VolScalar& deltaK() const { return deltaK_; }
    const VolScalar& deltaP() const { return deltaP_; }

    void advanceTime();

private:

    nf::RunTime& rt_;
    State state_ = State::initial;
    VolScalar& K_;    ///< stored in the field database (old time needed)
    VolScalar deltaK_; ///< (4)
    VolScalar deltaP_; ///< (6)
    SurfScalar KfOwn_;  ///< K interpolated with phi_own
    SurfScalar KfNei_;  ///< K interpolated with phi_nei
    SurfScalar flux_;   ///< phi_own K_o + phi_nei K_n
    fvcc::SurfaceInterpolation<NeoN::scalar> divKOwn_; ///< interpolation of 'div(phi_own,K)'
    fvcc::SurfaceInterpolation<NeoN::scalar> divKNei_; ///< interpolation of 'div(phi_nei,K)'
    // Viscous work, part of delta K: div(linear(devRhoReff & U) & Sf), devRhoReff = -tau.
    FaceGeometry geometry_;
    fvcc::VolumeField<NeoN::Tensor> gradU_;
    VolVector tauU_;                 ///< tau & U, tau = mu dev(grad(U) + grad(U)^T)
    SurfVector tauUf_;
    SurfScalar work_;                ///< linear(tau & U) & Sf
    VolScalar workCell_;             ///< div(linear(tau & U) & Sf)
    fvcc::SurfaceInterpolation<NeoN::Vec3> linearV_;

    void computeK(const MomentumOperators& momentum); ///< (5)
};

// * * * * * * * * * * * * * * * * * 4 * * * * * * * * * * * * * * * * * * //

/// Corrector: second runs of 2.2 (4.1, velocity) and 2.1 (4.2, density). Owns no fields.
class Corrector
{
public:

    enum class State
    {
        initial,
        updated,
        outdated
    };

    friend const char* toString(State s)
    {
        switch (s)
        {
        case State::initial:
            return "initial";
        case State::updated:
            return "updated";
        default:
            return "outdated";
        }
    }

    void update(Predictor& predictor, const PVC& pvc, const HybridScheme& hybrid);

    void reset();

    State state() const { return state_; }

private:

    State state_ = State::initial;
};

// * * * * * * * * * * * * * * * * * 1 * * * * * * * * * * * * * * * * * * //

/// Time step: Co (18), dt (17), advance of time and of the old-time levels.
class TimeStep
{
public:

    enum class State
    {
        initial,
        updated,
        outdated
    };

    friend const char* toString(State s)
    {
        switch (s)
        {
        case State::initial:
            return "initial";
        case State::updated:
            return "updated";
        default:
            return "outdated";
        }
    }

    /// Co^max, adjustTimeStep and maxDeltaT are taken from controlDict via the RunTime.
    /// Co is the cell Courant number (OpenFOAM's cellCourant); there is no other choice.
    explicit TimeStep(nf::RunTime& rt);

    /// Initialization step 9: Co and the initial dt (OpenFOAM's setInitialDeltaT).
    void initialize(
        Foam::Time& runTime, const HybridScheme& hybrid, const Continuity& continuity, const KNP& knp
    );

    /// Co, dt; advances the time; moves the fields to their old-time levels.
    void update(Foam::Time& runTime, const HybridScheme& hybrid, Predictor& predictor, PVC& pvc, MechanicalSources& sources);

    void reset();

    State state() const { return state_; }
    scalar Co() const { return Co_; }

private:

    nf::RunTime& rt_;
    State state_ = State::initial;
    bool printStatistics_;
    scalar Co_ = 0; ///< (18), maximum over the faces
    FaceGeometry geometry_;    ///< |Sf| for the characteristic Courant number
    SurfScalar deltaCoeffs_;   ///< 1/|d| for the characteristic Courant number
    SurfScalar charCo_;        ///< max(|ap|, |am|) deltaCoeffs/|Sf| dt on the faces

    /// Maximum over the internal and processor faces of max(|u + c|, |u - c|) dt/dx.
    scalar maxCharacteristicCo(const KNP& knp);

    void computeCo(const HybridScheme& hybrid, const Continuity& continuity);
};

// * * * * * * * * * * * * * * * * * 7 * * * * * * * * * * * * * * * * * * //

/// End of iteration: resets the states of all blocks to outdated. Owns no fields.
class EndOfIteration
{
public:

    explicit EndOfIteration(const nf::RunTime& rt);

    void update(
        TimeStep& timeStep,
        Predictor& predictor,
        PVC& pvc,
        Corrector& corrector,
        HybridScheme& hybrid,
        MechanicalSources& sources
    );

private:

    bool printStatistics_;
};

} // namespace hcs
