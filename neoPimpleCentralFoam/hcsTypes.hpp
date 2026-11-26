// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Matvey Kraposhin

#pragma once

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

#include "centralKernels.hpp"

#include <string>

namespace hcs
{

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

using NeoN::scalar;
using VolScalar = fvcc::VolumeField<NeoN::scalar>;
using VolVector = fvcc::VolumeField<NeoN::Vec3>;
using SurfScalar = fvcc::SurfaceField<NeoN::scalar>;
using SurfVector = fvcc::SurfaceField<NeoN::Vec3>;

/// A cell field with 'calculated' boundaries (processor patches exchange), filled with zero.
template<typename T = NeoN::scalar>
fvcc::VolumeField<T> makeVolField(const nf::RunTime& rt, const std::string& name)
{
    fvcc::VolumeField<T> f(
        rt.exec, name, rt.nfMesh, fvcc::createCalculatedProcBCs<fvcc::VolumeBoundary<T>>(rt.nfMesh)
    );
    NeoN::fill(f.internalVector(), NeoN::zero<T>());
    NeoN::fill(f.boundaryData().value(), NeoN::zero<T>());
    return f;
}

/// A face field with 'calculated' boundaries, filled with zero.
template<typename T = NeoN::scalar>
fvcc::SurfaceField<T> makeSurfField(const nf::RunTime& rt, const std::string& name)
{
    fvcc::SurfaceField<T> f(
        rt.exec, name, rt.nfMesh, fvcc::createCalculatedProcBCs<fvcc::SurfaceBoundary<T>>(rt.nfMesh)
    );
    NeoN::fill(f.internalVector(), NeoN::zero<T>());
    NeoN::fill(f.boundaryData().value(), NeoN::zero<T>());
    return f;
}

inline VolScalar makeVolScalar(const nf::RunTime& rt, const std::string& name)
{
    return makeVolField<NeoN::scalar>(rt, name);
}

inline SurfScalar makeSurfScalar(const nf::RunTime& rt, const std::string& name)
{
    return makeSurfField<NeoN::scalar>(rt, name);
}

/// Sf and |Sf| of the mesh as face fields.
struct FaceGeometry
{
    SurfVector Sf;
    SurfScalar magSf;

    explicit FaceGeometry(const nf::RunTime& rt)
        : Sf(makeSurfField<NeoN::Vec3>(rt, "Sf")), magSf(makeSurfScalar(rt, "magSf"))
    {
        kernels::faceGeometry(rt.nfMesh, Sf, magSf);
    }
};

/// A face field filled with a constant: +1 selects the owner side, -1 the neighbour side
/// in a flux-directed (upwind-biased) interpolation.
inline SurfScalar makeDirection(const nf::RunTime& rt, const std::string& name, scalar sign)
{
    SurfScalar f = makeSurfScalar(rt, name);
    NeoN::fill(f.internalVector(), sign);
    NeoN::fill(f.boundaryData().value(), sign);
    return f;
}

/// The interpolation scheme of fvSchemes/interpolationSchemes/<key> (or 'default').
template<typename T = NeoN::scalar>
fvcc::SurfaceInterpolation<T> interpolationScheme(const nf::RunTime& rt, const std::string& key)
{
    NF_ASSERT(
        rt.fvSchemesDict.contains("interpolationSchemes"),
        "fvSchemes has no interpolationSchemes dictionary"
    );
    const auto& schemes = rt.fvSchemesDict.subDict("interpolationSchemes");
    const std::string entry = schemes.contains(key) ? key : std::string("default");
    NF_ASSERT(schemes.contains(entry), "interpolationSchemes has no entry " << key);

    NeoN::TokenList tokens;
    if (schemes.isType<std::string>(entry))
    {
        const auto& word = schemes.get<std::string>(entry);
        NF_ASSERT(word != "none", "interpolationSchemes: add an explicit entry " << key);
        tokens = NeoN::TokenList({word});
    }
    else
    {
        tokens = schemes.get<NeoN::TokenList>(entry);
    }
    // The stored TokenList may have been read before; the factory reads from the cursor.
    tokens.reset();
    return fvcc::SurfaceInterpolation<T>(rt.exec, rt.nfMesh, tokens);
}

/// The interpolation of the divergence scheme fvSchemes/divSchemes/<key> ("Gauss <interpolation>").
template<typename T = NeoN::scalar>
fvcc::SurfaceInterpolation<T> divInterpolationScheme(const nf::RunTime& rt, const std::string& key)
{
    NF_ASSERT(rt.fvSchemesDict.contains("divSchemes"), "fvSchemes has no divSchemes dictionary");
    const auto& schemes = rt.fvSchemesDict.subDict("divSchemes");
    NF_ASSERT(schemes.contains(key), "divSchemes has no entry " << key);
    NeoN::TokenList tokens = schemes.get<NeoN::TokenList>(key);
    tokens.reset();
    const auto gauss = tokens.next<std::string>();
    NF_ASSERT(gauss == "Gauss", "divSchemes " << key << ": only Gauss is supported, got " << gauss);
    return fvcc::SurfaceInterpolation<T>(rt.exec, rt.nfMesh, tokens);
}

/// Central differencing, as OpenFOAM's linearInterpolate.
inline fvcc::SurfaceInterpolation<NeoN::scalar> linearScheme(const nf::RunTime& rt)
{
    return fvcc::SurfaceInterpolation<NeoN::scalar>(
        rt.exec, rt.nfMesh, NeoN::TokenList({std::string("linear")})
    );
}

/**
 * NeoN's DSL factories (dsl::imp::ddt, dsl::imp::div, dsl::exp::source, ...) take non-const
 * references, but the operators only read the fields they get, except for the solved one.
 * This is the one place where a field owned by another block is passed to them.
 */
template<typename FieldT>
FieldT& readOnlyOperand(const FieldT& field)
{
    return const_cast<FieldT&>(field);
}

/// Logs a state transition of a block: "<block>: <from> -> <to>".
template<typename StateT>
void logTransition(const std::string& block, StateT from, StateT to)
{
    NeoN::Logging::info("{}: {} -> {}", block, toString(from), toString(to));
}

} // namespace hcs
