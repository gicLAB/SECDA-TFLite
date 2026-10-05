Module.bazel uses:

    bazel_dep(name = "systemc_bazel", version = "2.3.3c")

Workspace uses:

    load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
    git_repository(
    name = "secda_tools",
    remote = "https://github.com/judeharis/secda_tools.git",
    commit = "4f14aa0c322034a4d5f69aa54e95f6fc2440a813",
    )   
