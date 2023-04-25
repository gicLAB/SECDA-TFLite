# - Find SYSC
# Find the native SYSC includes and library
#
#  SYSC_INCLUDES    - where to find fftw3.h
#  SYSC_LIBRARIES   - List of libraries when using SYSC.
#  SYSC_FOUND       - True if SYSC found.

if (SYSC_INCLUDES)
  # Already in cache, be silent
  set (SYSC_FIND_QUIETLY TRUE)
endif (SYSC_INCLUDES)

find_path (SYSC_INCLUDES systemc.h PATHS $ENV{SYSTEMC_HOME}/include)
find_library (SYSC_LIBRARIES NAMES systemc-2.3.3 systemc PATHS $ENV{SYSTEMC_HOME}/lib-linux64)


# handle the QUIETLY and REQUIRED arguments and set SYSC_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (SYSC DEFAULT_MSG SYSC_LIBRARIES SYSC_INCLUDES)

mark_as_advanced (SYSC_LIBRARIES SYSC_INCLUDES)