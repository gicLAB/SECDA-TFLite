add_subdirectory(driver)

if(NOT BUILD_ARM)
    include_directories(${acc_testing_SOURCE_DIR}/secda_tflite/)
    add_library(MMV4 acc.sc.cc)
    target_link_libraries(MMV4 PUBLIC api_sysc_v2)
    target_include_directories(MMV4 PUBLIC ${SYSC_INCLUDES})
    target_compile_options(MMV4 PUBLIC -fPIC -DSYSC)
endif()
