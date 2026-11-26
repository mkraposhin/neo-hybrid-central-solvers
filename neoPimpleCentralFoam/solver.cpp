// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#include "solver.hpp"

namespace hcs
{

Solver::Solver(Foam::Time& runTime, nf::RunTime& rt)
    : runTime_(runTime),
      rt_(rt),
      gas_(PerfectGas::read(rt.mesh)),
      fields_(fvcc::VectorCollection::instance(rt.db, "VectorCollection")),
      timeStep_(rt),
      pvc_(rt, fields_, boundaryContext_),
      predictor_(rt, fields_, gas_, boundaryContext_),
      hybrid_(rt),
      sources_(rt, fields_),
      endOfIteration_(rt),
      pimple_(rt.fvSolutionDict)
{
    // The fields the boundary conditions may need, under the names of the case dictionaries.
    const auto& thermo = predictor_.thermo();
    boundaryContext_.insert("phi", hybrid_.phi());
    boundaryContext_.insert("U", predictor_.momentum().U());
    boundaryContext_.insert("psi", thermo.psi());
    boundaryContext_.insert("thermo:psi", thermo.psi());
    boundaryContext_.insert("gamma", thermo.gammaField());
}

void Solver::initialize()
{
    NeoN::Logging::info("Initialization");

    auto& continuity = predictor_.continuity();
    auto& momentum = predictor_.momentum();
    auto& thermo = predictor_.thermo();
    auto& knp = pvc_.knp();
    auto& pressure = pvc_.pressure();

    thermo.initialize();                                          // 1
    continuity.initialize(thermo, pressure);                      // 2
    momentum.initialize(continuity);                              // 3
    hybrid_.initialize();                                         // 4
    knp.initialize(continuity, momentum, thermo, hybrid_);        // 5
    pressure.initialize(knp, continuity, momentum);               // 6
    hybrid_.update(pressure, knp, continuity, momentum, thermo);  // 7
    sources_.initialize(momentum);                                // 8
    timeStep_.initialize(runTime_, hybrid_, continuity, knp);     // 9
    endOfIteration_.update(                                       // 10
        timeStep_,
        predictor_,
        pvc_,
        corrector_,
        hybrid_,
        sources_
    );
}

void Solver::iterate(bool firstIteration)
{
    predictor_.update(firstIteration, pvc_.pressure(), pvc_.knp(), hybrid_, sources_); // 2
    pvc_.update(                                                                         // 3
        predictor_.continuity(),
        predictor_.momentum(),
        predictor_.thermo(),
        hybrid_
    );
    corrector_.update(predictor_, pvc_, hybrid_); // 4
    hybrid_.update(                               // 5
        pvc_.pressure(),
        pvc_.knp(),
        predictor_.continuity(),
        predictor_.momentum(),
        predictor_.thermo()
    );
    sources_.update(predictor_.momentum(), predictor_.continuity(), pvc_.pressure(), hybrid_); // 6
    endOfIteration_.update(timeStep_, predictor_, pvc_, corrector_, hybrid_, sources_);       // 7
}

void Solver::run()
{
    NeoN::Logging::info("Starting time loop");
    while (runTime_.run())
    {
        timeStep_.update(runTime_, hybrid_, predictor_, pvc_, sources_); // 1

        nf::ResidualMap residuals;
        while (pimple_.loop(residuals))
        {
            NeoN::Logging::info("PIMPLE outer iteration");
            iterate(pimple_.firstIter());
        }

        // centralContinuityErrs.H: the cumulative error grows by the global error of the last
        // outer iteration.
        const scalar cumulative = predictor_.continuity().accumulateContinuityError();
        if (printStatistics(rt_))
        {
            NeoN::Logging::info("    cumulative continuity error = {}", cumulative);
        }

        if (runTime_.writeTime())
        {
            write();
        }
        runTime_.printExecutionTime(Foam::Info);
    }
}

void Solver::write() const
{
    pvc_.pressure().write();
    predictor_.momentum().write();
    predictor_.thermo().write();
    predictor_.continuity().write();
}

} // namespace hcs
