load("@rules_cc//cc:defs.bzl", "cc_library")
load("//:def.bzl", "secda_copts")

package(default_visibility = ["//visibility:public"])

common_copts = secda_copts()

cc_library(
    name = "accelerator",
    srcs = [
        "acc_config.sc.h",
    ] + select({
        "@secda_tools//:sysc": ["acc.sc.cc"],
        "//conditions:default": [],
    }),
    hdrs = select({
        "@secda_tools//:sysc": [
            "acc.sc.h",
            "compute.sc.h",
            "hwc.sc.h",
        ],
        "//conditions:default": [],
    }),
    copts = common_copts,
    deps = ["//third_party:secdav5"],
)
