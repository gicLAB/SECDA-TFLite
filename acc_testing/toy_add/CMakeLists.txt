add_subdirectory(driver)


if(NOT BUILD_ARM)
include_directories(${acc_testing_SOURCE_DIR}/secda_tflite/)
add_library(ToyAdd acc.sc.cc)
target_link_libraries(ToyAdd PUBLIC api_sysc_v2)
target_include_directories(ToyAdd PUBLIC ${SYSC_INCLUDES})
target_compile_options(ToyAdd PUBLIC -fPIC -DSYSC)
endif()

