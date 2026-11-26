// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#pragma once

#include "blocks.hpp"

#include "NeoFOAM/solutionControl/pimpleControl.hpp"

namespace hcs
{

/**
 * @brief Owns the blocks and executes them as an FCFS queue.
 *
 * Members are constructed in declaration order: the time step, then PVC (reads p),
 * then the predictor (reads U and T), then the remaining blocks.
 */
class Solver
{
public:

    Solver(Foam::Time& runTime, nf::RunTime& rt);

    /// Initialization steps 1-10.
    void initialize();

    /// The time loop.
    void run();

private:

    Foam::Time& runTime_;
    nf::RunTime& rt_;
    PerfectGas gas_;
    fvcc::VectorCollection& fields_;
    /// Non-owning: filled with fields of the blocks below after their construction.
    fvcc::BoundaryContext boundaryContext_;

    TimeStep timeStep_;
    PVC pvc_;
    Predictor predictor_;
    HybridScheme hybrid_;
    MechanicalSources sources_;
    Corrector corrector_;
    EndOfIteration endOfIteration_;

    nf::PimpleControl pimple_;

    /// One outer iteration: blocks 2-7.
    void iterate(bool firstIteration);

    void write() const;
};

} // namespace hcs
