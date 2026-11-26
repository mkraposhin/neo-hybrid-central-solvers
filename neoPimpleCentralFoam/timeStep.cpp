// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#include "blocks.hpp"
#include "centralKernels.hpp"

namespace hcs
{

TimeStep::TimeStep(nf::RunTime& rt)
    : rt_(rt), printStatistics_(printStatistics(rt)), geometry_(rt),
      deltaCoeffs_(makeSurfScalar(rt, "TimeStep::deltaCoeffs")),
      charCo_(makeSurfScalar(rt, "characteristicCo"))
{
    kernels::deltaCoeffs(rt.nfMesh, deltaCoeffs_);
    NeoN::Logging::info(
        "1 TimeStep: maxCo = {}, adjustTimeStep = {}, maxDeltaT = {}",
        rt.maxCo,
        rt.adjustTimeStep,
        rt.maxDeltaT
    );
}

void TimeStep::initialize(
    Foam::Time& runTime,
    const HybridScheme& hybrid,
    const Continuity& continuity,
    const KNP& knp
)
{
    computeCo(hybrid, continuity);

    // (17), OpenFOAM's setInitialDeltaT: never larger than the deltaT of controlDict.
    if (rt_.adjustTimeStep && Co_ > Foam::SMALL)
    {
        runTime.setDeltaT(
            Foam::min(rt_.maxCo * runTime.deltaTValue() / Co_, Foam::min(runTime.deltaTValue(), rt_.maxDeltaT))
        );
    }
    rt_.dt = runTime.deltaTValue();
    if (printStatistics_)
    {
        NeoN::Logging::info(
            "    initial deltaT: {}, max Co: {}, max characteristic Co: {}",
            rt_.dt,
            Co_,
            maxCharacteristicCo(knp)
        );
    }

    logTransition("1 TimeStep", state_, State::updated);
    state_ = State::updated;
}

scalar TimeStep::maxCharacteristicCo(const KNP& knp)
{
    kernels::characteristicCourant(knp.ap(), knp.am(), deltaCoeffs_, geometry_.magSf, rt_.dt, charCo_);
    return kernels::minMaxInternalAndProc(charCo_).max;
}

void TimeStep::update(
    Foam::Time& runTime,
    const HybridScheme& hybrid,
    Predictor& predictor,
    PVC& pvc,
    MechanicalSources& sources
)
{
    computeCo(hybrid, predictor.continuity());
    const scalar dtOld = runTime.deltaTValue();
    nf::syncRunTimes(runTime, rt_, Co_);

    ++runTime;
    rt_.t = runTime.value();
    rt_.dt = runTime.deltaTValue();
    NeoN::Logging::info("Time = {}, deltaT = {}", rt_.t, rt_.dt);
    if (printStatistics_)
    {
        // Co is proportional to dt: the maximum Courant number of the step being taken; the
        // characteristic Courant number is evaluated with the dt of that step.
        NeoN::Logging::info(
            "    max Co: {}, max characteristic Co: {}",
            Co_ * rt_.dt / dtOld,
            maxCharacteristicCo(pvc.knp())
        );
    }

    // The new time level makes the current values old: every field with a time derivative
    // moves to its old-time level. Each owner performs the move on its own fields.
    predictor.continuity().advanceTime();
    predictor.momentum().advanceTime();
    predictor.thermo().advanceTime();
    pvc.pressure().advanceTime();
    sources.advanceTime();

    logTransition("1 TimeStep", state_, State::updated);
    state_ = State::updated;
}

void TimeStep::reset() { state_ = State::outdated; }

void TimeStep::computeCo(const HybridScheme& hybrid, const Continuity& continuity)
{
    // (18): Co = max over the cells of 0.5 dt sum_f |phi_f| / (rho V), phi = phi_own + phi_nei.
    Co_ = kernels::maxCellCourant(hybrid.phiOwn(), hybrid.phiNei(), continuity.rho(), rt_.dt);
}

} // namespace hcs
