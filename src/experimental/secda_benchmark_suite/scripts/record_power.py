import cv2
import numpy as np
from mss import mss
from os import listdir
from os.path import isfile, join
import argparse

import signal
import sys


# write code to create new image each add two images together increasing the size of the merged image
def merge(screen, i2, x_offset, y_offset):
    screen[y_offset : y_offset + i2.shape[0], x_offset : x_offset + i2.shape[1]] = i2
    return screen


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Capture Experiment Video")
    parser.add_argument("name", type=str, help="name of the experiment")
    # parser.add_argument(
    #     "-n", "--name", type=str, required=True, help="name of the experiment"
    # )
    args = parser.parse_args(raw_args)

    # Output file
    name = parser.parse_args().name
    mypath = "./files/"
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    outname = mypath + f"{name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(outname, fourcc, 20.0, (1280, 960))

    # Capture from camera and screen
    cap = cv2.VideoCapture(-1)
    sct = mss()
    bounding_box = {"top": 2000, "left": 0, "width": 1920, "height": 1740}
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    def signal_handler(*args):
        cap.release()
        out.release()
        # cv2.destroyAllWindows()
        # print("Power Capture Successful!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # signal.signal(signal.SIGINT, handler)
    # signal.signal(signal.SIGINT, signal_handler)
    try:
        with open("/home/jude/Workspace/SECDA-TFLite_v1.2/src/secda_benchmark_suite/power/files/coords.txt", "r") as f:
            px, py, pxs, pys = [int(x) for x in f.read().split()]
    except:
        print("No coords.txt found")
        px, py, pxs, pys = (82, 298, 350, 80)


    while True:
        # Screen Capture
        screen = np.array(sct.grab(bounding_box))
        scale_percent = 50  # percent of original size
        width = int(screen.shape[1] * scale_percent / 100)
        height = int(screen.shape[0] * scale_percent / 100)
        dim = (width, height)
        screen = cv2.resize(screen, dim, interpolation=cv2.INTER_AREA)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen_S = screen.shape
        screen = np.concatenate(
            (screen, np.zeros((screen.shape[0], 200, 3), dtype=np.uint8)), axis=1
        )

        # Camera Capture
        ret, frame = cap.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        frame = frame[py : py + pys, px : px + pxs, :3]  # // adjust to focus on mWH

        scale_percent = 200  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Merge Screen and Camera
        frame = merge(screen, frame, screen_S[0] - 60, 20)

        # Write the frame
        frame = cv2.resize(frame, (1280, 960))
        out.write(frame)

        # print("Press Ctrl+C")
        # signal.pause()

        # # Display the resulting frame
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) == ord("q"):
        #     break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    # sys.exit(0)
