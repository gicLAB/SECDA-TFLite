import sys


file = sys.argv[1]
# open file called temp_out.txt
open_file = open(file, "r")

avg_error = 0
# find the line that contains the word "avg_error"
for line in open_file:
    if "avg_error" in line:
        avg_error = line.split("avg_error=")[1].split(",")[0]

avg_error = float(avg_error)
print("Average error: ", avg_error)
if avg_error < 0.01:
    # print("No errors found")
    sys.exit(0)
else:
    # print("Error found")
    sys.exit(1)
