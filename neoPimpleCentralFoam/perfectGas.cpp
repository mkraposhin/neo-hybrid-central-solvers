// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#include "perfectGas.hpp"

#include "IOdictionary.H"

namespace hcs
{

PerfectGas PerfectGas::read(const Foam::fvMesh& mesh)
{
    Foam::IOdictionary dict(Foam::IOobject(
        "thermophysicalProperties",
        mesh.time().constant(),
        mesh,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE,
        Foam::IOobject::NO_REGISTER
    ));

    PerfectGas gas {
        .Cp = dict.get<Foam::scalar>("Cp"),
        .R = dict.get<Foam::scalar>("R"),
        .mu = dict.get<Foam::scalar>("mu"),
        .Pr = dict.get<Foam::scalar>("Pr")
    };

    if (!(gas.R > 0 && gas.Cp > gas.R && gas.mu >= 0 && gas.Pr > 0))
    {
        FatalIOErrorInFunction(dict) << "Expected Cp > R > 0, mu >= 0 and Pr > 0; got Cp = "
                                     << gas.Cp << ", R = " << gas.R << ", mu = " << gas.mu
                                     << ", Pr = " << gas.Pr << Foam::exit(Foam::FatalIOError);
    }

    NeoN::Logging::info(
        "Perfect gas: Cp = {}, R = {}, mu = {}, Pr = {}", gas.Cp, gas.R, gas.mu, gas.Pr
    );
    return gas;
}

} // namespace hcs
