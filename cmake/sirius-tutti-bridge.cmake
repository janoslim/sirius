# FlowCSD: link the Tutti bridge (tools/flowcsd_tutti_bridge) into Sirius.
#
# SIRIUS_TUTTI_BRIDGE_DIR names the bridge's install directory (the .so, its
# header and Tutti's shared libraries; see
# motivational_studies/storage-gpu-query/sirius/03_build_tutti_bridge.sh).  Set
# it as a CMake variable or in the environment; the environment wins, so
# `SIRIUS_TUTTI_BRIDGE_DIR=... make` works on an existing build tree, and the
# value is cached for later reconfigures.  Without it Sirius builds as before and
# io::tutti::tutti_service::instance() throws "Sirius built without Tutti".
set(SIRIUS_TUTTI_BRIDGE_DIR
    ""
    CACHE PATH "Install directory of flowcsd_tutti_bridge (empty: no Tutti)")
if(DEFINED ENV{SIRIUS_TUTTI_BRIDGE_DIR})
  set(SIRIUS_TUTTI_BRIDGE_DIR
      "$ENV{SIRIUS_TUTTI_BRIDGE_DIR}"
      CACHE PATH "Install directory of flowcsd_tutti_bridge (empty: no Tutti)" FORCE)
endif()

if(SIRIUS_TUTTI_BRIDGE_DIR)
  set(_sirius_tutti_bridge_library
      "${SIRIUS_TUTTI_BRIDGE_DIR}/libflowcsd_tutti_bridge.so")
  if(NOT EXISTS "${_sirius_tutti_bridge_library}")
    message(
      FATAL_ERROR
        "SIRIUS_TUTTI_BRIDGE_DIR=${SIRIUS_TUTTI_BRIDGE_DIR} has no libflowcsd_tutti_bridge.so"
    )
  endif()
  message(STATUS "Sirius: linking the Tutti bridge ${_sirius_tutti_bridge_library}")

  # Only io/tutti/tutti_service.cpp includes the bridge header.
  target_include_directories(sirius_objects PRIVATE "${SIRIUS_TUTTI_BRIDGE_DIR}")
  target_compile_definitions(sirius_objects PRIVATE SIRIUS_WITH_TUTTI=1)

  # Both final libraries carry sirius_objects; the archive passes the bridge on
  # to whatever links it (the unit tests and benchmarks).
  foreach(_target sirius_core sirius_shared)
    target_link_libraries(${_target} PRIVATE "${_sirius_tutti_bridge_library}")
  endforeach()

  # The installed libsirius.so (RPATH $ORIGIN) must still find the bridge; the
  # bridge finds its Tutti libraries next to itself.
  set_property(
    TARGET sirius_shared
    APPEND
    PROPERTY INSTALL_RPATH "${SIRIUS_TUTTI_BRIDGE_DIR}")
endif()
