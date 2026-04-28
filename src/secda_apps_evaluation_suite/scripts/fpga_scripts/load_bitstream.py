#!/usr/bin/python3
from pynq import Overlay
import argparse
import os


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Wrapper to load valid bitstreams")

    parser.add_argument(
        dest="bitstream",
        type=str,
        default="/home/xilinx/pynq/overlays/axi4mlir_maps/mm4x4_v1_highv1_nostatus.bit",
        help="Path to the bitstream file",
    )

    # add quiet option
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args(raw_args)
    quiet = args.quiet

    if os.path.isfile(args.bitstream):
        if not quiet:
            print("Loading bitstream: {}".format(args.bitstream))
        overlay = Overlay(args.bitstream)
        if not quiet:
            print(overlay.ip_dict.keys())
    else:
        print("Error! Could not find file: {}".format(args.bitstream))
        return -1


if __name__ == "__main__":
    main()
