import sys
import pandas as pd
import json
import datetime
import os


def get_df_from_csv(csv_path):
    try:
        f = open(csv_path, "r")
    except:
        print(csv_path + " does not exist")
        exit(1)
    save = 0
    savedString = ""
    while True:
        line = f.readline()
        if save == 2:
            savedString += line
        if (
            line
            == "============================== Summary by node type ==============================\n"
        ):
            save += 1
        if not line:
            break
    node_table = savedString.split("\n\n")[0].split("\n")
    node_table = [x.replace(" ", "").split(",") for x in node_table]
    df = pd.DataFrame.from_records(node_table)
    df.columns = df.iloc[0]
    df = df[1:]
    return df


def add_new_columns(df, column, value):
    df[column] = value
    return df


def process_layer_details(df, run_dict, run_name):
    # runtime is combination of driver, del and hardware
    runtime = (
        run_dict["hardware"]
        + "_"
        + run_dict["acc_version"]
        + "_"
        + run_dict["del"]
        + "_"
        + run_dict["del_version"]
    )
    df = add_new_columns(df, "Runtime", runtime)
    # get sum of avg_ms
    run_dict["total_latency"] = int(df["avg_ms"].astype(float).sum()* 1000) # convert to us
    run_dict["acc_layer"] = int(
        df[df["nodetype"].str.lower().str.contains("del")]["avg_ms"].astype(float).sum() * 1000 
    )
    run_dict["cpu_layers"] = int(run_dict["total_latency"] - run_dict["acc_layer"])
    # # save df to csv
    # os.makedirs(f"results/{now}", exist_ok=True)
    # df.to_csv(f"results/{now}/" + run_name + ".csv")
    return run_dict


def gather_run_info(run_dict, run_name):
    # open file called prf.csv and read it line by line
    if run_dict["valid"] == "1":
        acc_prf = {}
        if run_dict["hardware"] != "CPU":
            # open file called prf.csv, handle if it does not exist
            try:
                open_file = open(f"tmp/{run_name}_prf.csv", "r")
                for line in open_file:
                    # separate each line by comma
                    line = line.replace("\n", "").split(",")
                    acc_prf[line[0]] = line[1]
            except:
                print(f"tmp/{run_name}_prf.csv" + " does not exist")

        run_dict = {**run_dict, **acc_prf}
        run_dict = process_layer_details(
            get_df_from_csv(f"tmp/{run_name}_layer.csv"), run_dict, run_name
        )
    else:
        run_dict["total_latency"] = 0
        run_dict["acc_layer"] = 0
        run_dict["cpu_layers"] = 0

    # save dict to json
    os.makedirs(f"results/{now}", exist_ok=True)
    with open(f"results/{now}/" + run_name + ".json", "w") as fp:
        json.dump(run_dict, fp)


args = sys.argv[1:]
run_dict = {}
run_dict["model"] = args[0]
run_dict["thread"] = args[1]
run_dict["num_run"] = args[2]
run_dict["hardware"] = args[3]
run_dict["acc_version"] = args[4]
run_dict["del"] = args[5]
run_dict["del_version"] = args[6]
run_dict["valid"] = args[7]
now = args[8]

# create run name from run_dict
run_name = "_".join(
    [
        run_dict["hardware"],
        run_dict["acc_version"],
        run_dict["del"],
        run_dict["del_version"],
        run_dict["model"],
        run_dict["thread"],
        run_dict["num_run"],
    ]
)
gather_run_info(run_dict, run_name)
