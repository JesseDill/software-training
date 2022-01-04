# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_mineral_deposit_tracking_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED mineral_deposit_tracking_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(mineral_deposit_tracking_FOUND FALSE)
  elseif(NOT mineral_deposit_tracking_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(mineral_deposit_tracking_FOUND FALSE)
  endif()
  return()
endif()
set(_mineral_deposit_tracking_CONFIG_INCLUDED TRUE)

# output package information
if(NOT mineral_deposit_tracking_FIND_QUIETLY)
  message(STATUS "Found mineral_deposit_tracking: 0.0.0 (${mineral_deposit_tracking_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'mineral_deposit_tracking' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${mineral_deposit_tracking_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(mineral_deposit_tracking_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${mineral_deposit_tracking_DIR}/${_extra}")
endforeach()
