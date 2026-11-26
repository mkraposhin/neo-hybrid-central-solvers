# SPDX-License-Identifier: Unlicense
#
# SPDX-FileCopyrightText: 2026 Matvey Kraposhin

add_executable(
  neoPimpleCentralFoam
  neoPimpleCentralFoam.cpp
  solver.cpp
  perfectGas.cpp
  timeStep.cpp
  continuity.cpp
  momentumOperators.cpp
  energyThermo.cpp
  knp.cpp
  pressureEquation.cpp
  hybridScheme.cpp
  mechanicalSources.cpp
  centralKernels.cpp)
# The NeoFOAM target passes namespaceFoam=1 to its users, so fvCFD.H does not inject
# 'using namespace Foam': all OpenFOAM names below are qualified.
target_link_libraries(neoPimpleCentralFoam NeoFOAM)
install(TARGETS neoPimpleCentralFoam)
