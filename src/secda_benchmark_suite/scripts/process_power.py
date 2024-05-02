import argparse
import signal
import sys

import cv2
import cupy as np
import csv
import pytesseract
import easyocr


def clean_square_for_OCR(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            cv2.drawContours(opening, [c], -1, 0, -1)

    result = 255 - opening
    result = opening
    result = cv2.GaussianBlur(result, (5, 5), 0)
    return result


def main(raw_args=None):
    reader = easyocr.Reader(["en"])
    parser = argparse.ArgumentParser(description="Capture Experiment Video")
    parser.add_argument("name", type=str, help="name of the experiment")
    parser.add_argument("exps", type=int, help="number of experiments")

    mypath = "./files/"
    run = "./tmp/runs.csv"
    name = parser.parse_args().name
    inname = mypath + f"{name}.mp4"
    otname = mypath + f"{name}_clean.mp4"
    exp_expected = parser.parse_args().exps

    cap = cv2.VideoCapture(inname)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 20.0
    size = (1280, 960)
    out = cv2.VideoWriter(otname, fourcc, fps, size)

    runfile = open(run, "r")
    runreader = csv.reader(runfile)
    runlist = list(runreader)
    csvfile = open(mypath + f"{name}_clean.csv", "w")
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        [
            "model",
            "hw",
            "version",
            "delegate",
            "del_version",
            "thread",
            "num_runs",
            "duration",
            "start_power",
            "end_power",
        ]
    )

    def signal_handler(*args):
        cap.release()
        out.release()
        csvfile.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    exp_count = 0
    curr_in_start = False
    curr_in_power = ""
    while True:
        if exp_count == exp_expected:
            break
        enter_frame = False
        exit_frame = False
        ret, frame = cap.read()
        if not ret:
            break

        check = frame[885:910, 20:230, :]
        check = cv2.resize(check, (check.shape[1] * 4, check.shape[0] * 4))
        ncheck = check

        # Check for start and end of run
        ocr = reader.readtext(ncheck)
        ocr = " ".join([x[1] for x in ocr])
        ocr = ocr.lower()
        check = cv2.putText(
            img=np.copy(ncheck),
            text=ocr,
            org=(0, 10),
            fontFace=2,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=1,
        )

        keyword_check = True if "start" in ocr and "power" in ocr else False
        if keyword_check and not curr_in_start:
            curr_in_start = True
            enter_frame = True

        elif not keyword_check and curr_in_start:
            curr_in_start = False
            exit_frame = True
            exp_count += 1


        ## OCR for power
        power = frame[20:100, 920:-135, :]  # y , x // adjust to focus on mWH
        power = cv2.resize(power, (power.shape[1] * 4, power.shape[0] * 4))
        npower = clean_square_for_OCR(power)
        # cv2.imshow('npower', npower)
        # if cv2.waitKey(1) == ord('q'):
        #     break

        # Don't save run frames except for the start and end
        if not enter_frame and not exit_frame:
            continue

        power_ocr = pytesseract.image_to_string(
            npower, lang="eng", config="--psm 8 -c tessedit_char_whitelist=0123456789"
        ).replace("\n\x0c", "")
        power_ocr = int(power_ocr) if power_ocr else 0

        if enter_frame:
            curr_in_power = power_ocr
        if exit_frame:
            row = runlist[exp_count - 1] + [curr_in_power, power_ocr]
            csvwriter.writerow(row)
        out.write(frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break

    # Release resources and close the video writer
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    csvfile.close()


if __name__ == "__main__":
    main()
