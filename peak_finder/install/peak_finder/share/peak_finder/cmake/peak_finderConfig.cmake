# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_peak_finder_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED peak_finder_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(peak_finder_FOUND FALSE)
  elseif(NOT peak_finder_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(peak_finder_FOUND FALSE)
  endif()
  return()
endif()
set(_peak_finder_CONFIG_INCLUDED TRUE)

# output package information
if(NOT peak_finder_FIND_QUIETLY)
  message(STATUS "Found peak_finder: 0.0.0 (${peak_finder_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'peak_finder' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${peak_finder_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(peak_finder_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${peak_finder_DIR}/${_extra}")
endforeach()
