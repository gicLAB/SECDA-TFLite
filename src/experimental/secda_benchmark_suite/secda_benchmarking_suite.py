import os
import subprocess
import sys
import json
import argparse
from datetime import datetime

# Load configuration from config.json
with open('../../config.json') as config_file:
    config = json.load(config_file)

board_user = config['board_user']
board_hostname = config['board_hostname']
board_dir = config['board_dir']
board_port = config['board_port']
conda_path = config['conda_path']
bench_dir = os.path.join(board_dir, 'benchmark_suite')

def help_function():
    print("""
    Usage: python3 secda_benchmark_suite.py -s -b -c -p -t -i -n name
    -s Skip running experiment
    -b Generate binaries
    -c Skip inference difference checks
    -p Power collection
    -t Test run
    -i Initialize the board
    -n Name of the experiment
    """)
    sys.exit(1)

def create_dir():
    subprocess.run(f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} 'mkdir -p {bench_dir} && mkdir -p {board_dir}/bitstreams && mkdir -p {bench_dir}/bins && mkdir -p {bench_dir}/models'", shell=True)
    subprocess.run(f"rsync -r -avz -e 'ssh -p {board_port}' ./model_gen/models {board_user}@{board_hostname}:{bench_dir}/", shell=True)
    subprocess.run(f"rsync -r -avz -e 'ssh -p {board_port}' ./bitstreams {board_user}@{board_hostname}:{board_dir}/", shell=True)
    subprocess.run(f"rsync -q -r -avz -e 'ssh -p {board_port}' ./scripts/fpga_scripts/ {board_user}@{board_hostname}:{board_dir}/scripts/", shell=True)
    print("Initialization Done")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action='store_true', help='Skip running experiment')
    parser.add_argument('-b', action='store_true', help='Generate binaries')
    parser.add_argument('-c', action='store_true', help='Skip inference difference checks')
    parser.add_argument('-p', action='store_true', help='Power collection')
    parser.add_argument('-t', action='store_true', help='Test run')
    parser.add_argument('-i', action='store_true', help='Initialize the board')
    parser.add_argument('-n', type=str, help='Name of the experiment')
    args = parser.parse_args()

    skip_bench = args.s
    bin_gen = args.b
    skip_inf_diff = args.c
    collect_power = args.p
    test_run = args.t
    init = args.i
    name = args.n if args.n else f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"

    def ctrl_c_handler(signum, frame):
        print("Exiting")
        sys.exit(1)

    import signal
    signal.signal(signal.SIGINT, ctrl_c_handler)

    print("-----------------------------------------------------------")
    print("-- SECDA-TFLite Benchmark Suite --")
    print("-----------------------------------------------------------")
    print("Configurations")
    print("--------------")
    print(f"Board User: {board_user}")
    print(f"Board Hostname: {board_hostname}")
    print(f"Board Benchmark Dir: {bench_dir}")
    print(f"Skip Bench: {skip_bench}")
    print(f"Bin Gen: {bin_gen}")
    print(f"Skip Inf Diff: {skip_inf_diff}")
    print(f"Collect Power: {collect_power}")
    print(f"Test Run: {test_run}")
    print(f"Name: {name}")
    print("-----------------------------------------------------------")

    if not skip_bench:
        print("Clearing cache")
        subprocess.run("rm -rf ./tmp", shell=True)

    if init:
        create_dir()

    print("-----------------------------------------------------------")
    print("Configuring Benchmark")
    print("-----------------------------------------------------------")
    subprocess.run(f"python3 scripts/configure_benchmark.py {int(bin_gen)}", shell=True)

    if bin_gen:
        print("-----------------------------------------------------------")
        print("Generating Binaries")
        subprocess.run("source ./generated/gen_bins.sh", shell=True)
        print("-----------------------------------------------------------")

    subprocess.run("source ./generated/configs.sh", shell=True)
    length = len(os.environ.get('hw_array', '').split())

    subprocess.run(f"source {conda_path}/activate tf", shell=True)

    if not skip_bench:
        print("-----------------------------------------------------------")
        print("Transferring Experiment Configurations to Target Device")
        subprocess.run(f"rsync -q -r -avz -e 'ssh -p {board_port}' ./generated/configs.sh {board_user}@{board_hostname}:{bench_dir}/", shell=True)
        subprocess.run(f"rsync -q -r -avz -e 'ssh -p {board_port}' ./generated/run_collect.sh {board_user}@{board_hostname}:{bench_dir}/", shell=True)
        subprocess.run(f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} 'cd {bench_dir}/ && chmod +x ./*.sh'", shell=True)

        if collect_power:
            print("-----------------------------------------------------------")
            print("Initializing Power Capture")
            subprocess.Popen(f"python3 scripts/record_power.py {name}", shell=True)
            with open('/tmp/record_power.py.pid', 'w') as pid_file:
                pid_file.write(str(os.getpid()))
            print("-----------------------------------------------------------")

        print("-----------------------------------------------------------")
        print("Running Experiments")
        print("-----------------------------------------------------------")
        subprocess.run(f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} 'cd {bench_dir}/ && ./run_collect.sh {int(process_on_fpga)} {int(skip_inf_diff)} {int(collect_power)} {int(test_run)}'", shell=True)

        if not test_run:
            print("Transferring Results to Host")
            subprocess.run(f"rsync --mkpath -q -r -av -e 'ssh -p {board_port}' {board_user}@{board_hostname}:{bench_dir}/tmp ./", shell=True)
            subprocess.run(f"rsync --mkpath -q -r -av -e 'ssh -p {board_port}' {board_user}@{board_hostname}:{bench_dir}/tmp ./tmp/{name}/", shell=True)
        print("-----------------------------------------------------------")

        if collect_power:
            if os.path.exists('/tmp/record_power.py.pid'):
                with open('/tmp/record_power.py.pid') as pid_file:
                    os.kill(int(pid_file.read()), signal.SIGTERM)
                print("-----------------------------------------------------------")
                print("Power Capture Done")
                print("-----------------------------------------------------------")
                print("Processing Power Data")
                print("-----------------------------------------------------------")
                subprocess.run(f"python3 scripts/process_power.py {name} {length}", shell=True)
                print("Power Processing Done")
                print("-----------------------------------------------------------")
            else:
                print("/tmp/record_power.py.pid not found")
            print(f"Simple csv: ./files/{name}_clean.csv")
            print("-----------------------------------------------------------")

    if not test_run:
        print("-----------------------------------------------------------")
        print("Post Processing")
        prev_failed = 0
        prev_hw = ""
        length = len(os.environ.get('hw_array', '').split())
        for i in range(length):
            index = i + 1
            HW = os.environ.get(f'hw_array[{i}]')
            MODEL = os.environ.get(f'model_array[{i}]')
            THREAD = os.environ.get(f'thread_array[{i}]')
            NUM_RUN = os.environ.get(f'num_run_array[{i}]')
            VERSION = os.environ.get(f'version_array[{i}]')
            DEL_VERSION = os.environ.get(f'del_version_array[{i}]')
            DEL = os.environ.get(f'del_array[{i}]')
            prev_failed = 0
            prev_hw = HW
            runname = f"{HW}_{VERSION}_{DEL}_{DEL_VERSION}_{MODEL}_{THREAD}_{NUM_RUN}"
            if i % 1 == 0:
                print("========================================================================")
                print(f"{runname} {index}/{length}")
                print("========================================================================")

            valid = 1
            if HW != "CPU" and not skip_inf_diff:
                if not os.path.exists(f'tmp/{runname}_id.txt'):
                    valid = 0
                    print(f"Correct Check File Missing for: {runname}")
                else:
                    subprocess.run(f"python3 scripts/fpga_scripts/check_valid.py tmp/{runname}_id.txt", shell=True)
                    if subprocess.returncode != 0:
                        valid = 0
                        print(f"Correctness Check Failed {runname}")

            subprocess.run(f"python3 scripts/process_run.py {MODEL} {THREAD} {NUM_RUN} {HW} {VERSION} {DEL} {DEL_VERSION} {valid} {name}", shell=True)
            if subprocess.returncode != 0:
                prev_failed = 1
                print("Process Run Failed")
                continue

        subprocess.run(f"python3 scripts/process_all_runs.py {name}", shell=True)
        print("-----------------------------------------------------------")
        print("Post Processing Done")
        print("-----------------------------------------------------------")

    print("-----------------------------------------------------------")
    print("Exiting SECDA-TFLite Benchmark Suite")
    print("-----------------------------------------------------------")

if __name__ == "__main__":
    main()