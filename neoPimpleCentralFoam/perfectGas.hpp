// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#pragma once

#include "hcsTypes.hpp"

#include "fvMesh.H"

#include <Kokkos_Core.hpp>

namespace hcs
{

/**
 * @brief Constant gas properties: the input data C_p, R, mu, Pr, and the pointwise relations
 * that depend on them only.
 *
 * Plain old data, so device kernels capture it by value. All member functions are callable on
 * the host and on the device.
 */
struct PerfectGas
{
    scalar Cp;
    scalar R;
    scalar mu;
    scalar Pr;

    /// (8): C_v = C_p - R.
    KOKKOS_INLINE_FUNCTION scalar Cv() const { return Cp - R; }

    /// (9): gamma = C_p / C_v.
    KOKKOS_INLINE_FUNCTION scalar gamma() const { return Cp / Cv(); }

    /// h(T), sensible enthalpy with the reference state T = 0, h = 0.
    KOKKOS_INLINE_FUNCTION scalar h(scalar T) const { return Cp * T; }

    /// dh/dT at temperature T.
    KOKKOS_INLINE_FUNCTION scalar dhdT(scalar /*T*/) const { return Cp; }

    /**
     * (10): T = T(h), the inverse of h(T), by Newton iterations starting from Tguess.
     * For a constant C_p the first iteration is already exact.
     */
    KOKKOS_INLINE_FUNCTION scalar T(scalar hValue, scalar Tguess) const
    {
        constexpr int maxIter = 20;
        constexpr scalar relTol = 1e-12;
        scalar Tk = Tguess;
        for (int k = 0; k < maxIter; ++k)
        {
            const scalar dT = (h(Tk) - hValue) / dhdT(Tk);
            Tk -= dT;
            if (Kokkos::fabs(dT) <= relTol * Kokkos::fabs(Tk))
            {
                break;
            }
        }
        return Tk;
    }

    /// (29): psi = 1/(R T).
    KOKKOS_INLINE_FUNCTION scalar psi(scalar T) const { return 1.0 / (R * T); }

    /// (15): c = sqrt(gamma R T).
    KOKKOS_INLINE_FUNCTION scalar c(scalar T) const { return Kokkos::sqrt(gamma() * R * T); }

    /// Laminar thermal diffusivity for h: alpha_eff = mu / Pr.
    KOKKOS_INLINE_FUNCTION scalar alphaEff() const { return mu / Pr; }

    /// Reads the flat dictionary constant/thermophysicalProperties (keys Cp, R, mu, Pr).
    static PerfectGas read(const Foam::fvMesh& mesh);
};

} // namespace hcs
