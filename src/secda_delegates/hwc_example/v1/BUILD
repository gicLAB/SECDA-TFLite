load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//:def.bzl", "secda_copts")

package(default_visibility = ["//visibility:public"])

common_copts = secda_copts()

cc_binary(
    name = "exp",
    srcs = ["experiment.cc"],
    copts = common_copts,
    deps = [
        "//experiments/vadd_exp/v1/accelerator/driver",
    ],
)
