import pandas as pd
import json
import glob
import sys
import os


# get date and time
args = sys.argv[1:]
now = args[0]

# get all json files from out folder
json_files = glob.glob(f"results/{now}/*.json")
json_files.sort(key=lambda x: os.path.getmtime(x))
runs = []
if len(json_files) == 0:
    print("No json files found")
    exit(1)
for json_file in json_files:
    with open(json_file, "r") as f:
        runs.append(json.load(f))

common_keys = set.intersection(*map(set, runs))
# find all the unique keys
unique_keys = set.union(*map(set, runs)) - common_keys

# create csv file with common keys as
common_keys = list(common_keys)
unique_keys = list(unique_keys)
cols = common_keys + unique_keys
df = pd.DataFrame(columns=cols)
for run in runs:
    df = pd.concat([df, pd.DataFrame(run, index=[0])], ignore_index=True)

# specialised ordering of columns
ordered_keys = [
    "model",
    "thread",
    "num_run",
    "hardware",
    "acc_version",
    "del",
    "del_version",
    "valid",
    "acc_layer",
    "cpu_layers",
    "total_latency",
]
unique_keys = unique_keys + [key for key in common_keys if key not in ordered_keys]
final_keys = ordered_keys + unique_keys
# re-order columns
df = df[final_keys]

# save df to csv
df.to_csv(f"results/benchmark_summary_{now}.csv")
df.to_csv(f"results/latest.csv")
print("Benchmark summary saved to results/benchmark_summary_{}.csv".format(now))
