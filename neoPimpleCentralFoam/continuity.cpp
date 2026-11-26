// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#include "blocks.hpp"

namespace hcs
{

namespace
{

/// a = b + c, element-wise.
void sum(NeoN::Vector<scalar>& a, const NeoN::Vector<scalar>& b, const NeoN::Vector<scalar>& c)
{
    auto [av, bv, cv] = NeoN::views(a, b, c);
    NeoN::parallelFor(
        a.exec(),
        {0, static_cast<NeoN::localIdx>(av.size())},
        NEON_LAMBDA(const NeoN::localIdx i) { av[i] = bv[i] + cv[i]; },
        "hcs::sum"
    );
}

/// a = b * c, element-wise.
void product(NeoN::Vector<scalar>& a, const NeoN::Vector<scalar>& b, const NeoN::Vector<scalar>& c)
{
    auto [av, bv, cv] = NeoN::views(a, b, c);
    NeoN::parallelFor(
        a.exec(),
        {0, static_cast<NeoN::localIdx>(av.size())},
        NEON_LAMBDA(const NeoN::localIdx i) { av[i] = bv[i] * cv[i]; },
        "hcs::product"
    );
}

} // namespace

Continuity::Continuity(nf::RunTime& rt, fvcc::VectorCollection& fields)
    : rt_(rt),
      printStatistics_(printStatistics(rt)),
      ofRho_(
          Foam::IOobject(
              "rho",
              rt.mesh.time().timeName(),
              rt.mesh,
              Foam::IOobject::NO_READ,
              Foam::IOobject::NO_WRITE
          ),
          rt.mesh,
          Foam::dimensionedScalar(Foam::dimDensity, 0),
          Foam::word("calculated")
      ),
      rho_(nf::constructAndRegister(fields, rt, ofRho_, false)),
      phi_(makeSurfScalar(rt, "phi")),
      rhoT_(makeVolScalar(rt, "rhoT")),
      solver_(rho_, rt)
{}

void Continuity::initialize(const EnergyThermo& thermo, const PressureEquation& pressure)
{
    close(thermo, pressure, false);
}

void Continuity::update(
    const EnergyThermo& thermo,
    const PressureEquation& pressure,
    const HybridScheme& hybrid
)
{
    if (pressure.state() == PressureEquation::State::solved)
    {
        close(thermo, pressure, printStatistics_);
    }
    else
    {
        // (1) with the blended fluxes: blending moves flux between the owner and neighbour
        // parts but keeps their sum.
        transport(hybrid.phiOwn(), hybrid.phiNei());
        logTransition("2.1 Continuity", state_, State::transported);
        state_ = State::transported;
    }
}

void Continuity::reset() { state_ = State::outdated; }

void Continuity::advanceTime() { fvcc::rotateOldTimes(rho_); }

void Continuity::write() const { nf::write(rho_, rt_.mesh); }

scalar Continuity::accumulateContinuityError()
{
    cumulativeContErr_ += globalContErr_;
    return cumulativeContErr_;
}

void Continuity::transport(const SurfScalar& phiA, const SurfScalar& phiB)
{
    // (1): ddt(rho) + div(phi) = 0, phi = phiA + phiB.
    sum(phi_.internalVector(), phiA.internalVector(), phiB.internalVector());
    sum(phi_.boundaryData().value(), phiA.boundaryData().value(), phiB.boundaryData().value());

    nf::PDE<NeoN::scalar> rhoEqn(NeoN::dsl::imp::ddt(rho_) + NeoN::dsl::exp::div(phi_));
    solver_.solve(rhoEqn);
    rho_.correctBoundaryConditions();
}

void Continuity::close(
    const EnergyThermo& thermo, const PressureEquation& pressure, const bool diagnose
)
{
    if (diagnose)
    {
        // massEqn.H with the new fluxes (19), (20), then centralContinuityErrs.H.
        transport(pressure.phiOwn(), pressure.phiNei());
        product(rhoT_.internalVector(), thermo.psi().internalVector(), pressure.p().internalVector());
        const auto& V = rt_.nfMesh.cellVolumes();
        const scalar totalMass = kernels::domainIntegrate(rho_.internalVector(), V);
        const scalar sumLocalContErr =
            kernels::domainIntegrateDifference(rho_.internalVector(), rhoT_.internalVector(), V, true)
            / totalMass;
        globalContErr_ =
            kernels::domainIntegrateDifference(rho_.internalVector(), rhoT_.internalVector(), V, false)
            / totalMass;
        NeoN::Logging::info(
            "    continuity errors: sum local = {}, global = {}", sumLocalContErr, globalContErr_
        );
    }

    // (11): rho = psi p, in the cells and on the boundary faces.
    product(rho_.internalVector(), thermo.psi().internalVector(), pressure.p().internalVector());
    product(rho_.boundaryData().value(),
            thermo.psi().boundaryData().value(),
            pressure.p().boundaryData().value());

    logTransition("2.1 Continuity", state_, State::closed);
    state_ = State::closed;
}

} // namespace hcs
