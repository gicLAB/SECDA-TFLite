load("@rules_cc//cc:defs.bzl", "cc_library")
load("//:def.bzl", "secda_copts")

package(default_visibility = ["//visibility:public"])

common_copts = secda_copts()

cc_library(
    name = "driver",
    srcs = [
        "acc_container.h",
        "driver.h",
        "systemc_binding.h",
    ],
    copts = common_copts,
    deps = [
        "//experiments/vadd_exp/v1/accelerator",
    ],
)
