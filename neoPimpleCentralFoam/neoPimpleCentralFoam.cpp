// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

/*---------------------------------------------------------------------------*\
Application
    neoPimpleCentralFoam

Description
    Pressure-based solver for compressible flows of a perfect gas with the
    hybrid Kurganov-Noelle-Petrova / PIMPLE method (pimpleCentralFoam),
    ported to NeoFOAM/NeoN.

    Step 1 of the port: construction of all blocks, their state machines and
    the queue of blocks; the relations are not computed yet.
\*---------------------------------------------------------------------------*/

#include "solver.hpp"

int main(int argc, char* argv[])
{
    Foam::argList::addOption("executor", "word", "NeoN executor type (Serial/CPU/GPU/default)");
#include "addCheckCaseOptions.H"
#include "setRootCase.H"
#include "createTime.H"
    NeoN::initialize(argc, argv);
    {
        auto rt = NeoFOAM::createAdapterRunTime(runTime, args);

        NeoFOAM::createMappedFvSolutionDicts(rt);
        rt.fvSchemesDict = NeoFOAM::mapFvSchemes(rt.fvSchemesDict);

        hcs::Solver solver(runTime, rt);
        solver.initialize();
        solver.run();
    }
    NeoN::finalize();

    Foam::Info << "End\n" << Foam::endl;
    return 0;
}

// ************************************************************************* //
