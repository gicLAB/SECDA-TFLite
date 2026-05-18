
from os import listdir
from os.path import isfile, join
import argparse
import signal
import sys
import datetime




def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Capture Experiment Video")
    parser.add_argument("name", type=str, help="name of the experiment")
    args = parser.parse_args(raw_args)

    # Output file
    name = parser.parse_args().name
    mypath = "../results/"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Timestamp: {timestamp}")



if __name__ == "__main__":
    main()
    # sys.exit(0)
