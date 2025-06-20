import sys

sys.dont_write_bytecode = True

import os
from string import Template
import json

hw_gen_tpl = "hw_gen.tpl.sh"


# open c++ header file and parse all the #define and their values into a dictionary
def parse_defines(file_name="test.h"):
    with open(file_name) as f:
        defines = {}
        for line in f:
            if line.startswith("#define"):
                tokens = line.split()
                if "BITS" in tokens[1]:
                    continue
                tokens[1] = tokens[1].replace("XVM_INT8_V3_0_SLV0_ADDR_", "")
                tokens[1] = tokens[1].replace("_DATA", "")
                defines[tokens[1]] = int(tokens[2], 16)
        # print(defines)
        print(f"rm = {defines}")


# print bitmap for an integer , put space between every 8 bits
def print_bitmap(num):
    s = bin(num)[2:]
    s = s.zfill(32)
    for i in range(0, len(s), 8):
        print(s[i : i + 8], end=" ")
    print()


def load_config(config_file):
    try:
        with open(config_file) as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: The file {config_file} was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file {config_file} is not a valid JSON file.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def find_hw_config(dir, hw):
    # check if the file.json exists in the directory
    for file in os.listdir(dir):
        if file.endswith(".json"):
            if file == hw + ".json":
                return dir + "/" + file
    # check subdirectories
    for subdir in os.listdir(dir):
        if os.path.isdir(dir + "/" + subdir):
            for file in os.listdir(dir + "/" + subdir):
                if file.endswith(".json"):
                    if file == hw + ".json":
                        return dir + "/" + subdir + "/" + file
    return


class mt(Template):
    delimiter = "&"
    idpattern = r"[a-z][_a-z0-9]*"


def dict_default(dict, para, default):
    return dict[para] if para in dict else default


class hardware_exp:
    def __init__(
        self,
        config,
        sc,
    ):
        self.sc = sc
        self.config = config
        self.board = dict_default(config, "board", "Z1")
        self.bconfig = sc["boards"][self.board]
        # self.board = config["board"]
        self.hw_link_dir = f"{sc['secda_tflite_path']}/{sc['hw_link_dir']}/"
        # self.vp = self.bconfig["vivado_path"]
        # self.vp_hls = self.bconfig["hls_path"]
        self.vp_hls = self.sc["vivado_2019_path"]
        self.vp = (self.sc["vivado_2024_path"]
            if self.bconfig["hlx_version"] == "2024"
            else self.sc["vivado_2019_path"])

        self.svp_hls = sc["remote_server"]["server_2019_path"]
        self.svp_hlx = (
            sc["remote_server"]["server_2024_path"]
            if self.bconfig["hlx_version"] == "2024"
            else sc["remote_server"]["server_2019_path"]
        )

        self.gateway = sc["remote_server"]["gateway"]

        # accelerator stuff
        self.acc_name = config["acc_name"]
        self.acc_version = config["acc_version"]
        self.acc_sub_version = config["acc_sub_version"]
        self.acc_test = config["acc_test"]
        self.acc_test_id = ("_" + self.acc_test_id) if self.acc_test else ""
        self.acc_test_desc = config["acc_test_desc"]
        self.acc_link_folder = os.path.abspath(
            self.hw_link_dir + config["acc_link_folder"]
        )

        # hardware stuff
        self.fpga_part = self.bconfig["fpga_part"]
        self.top = config["top"]
        self.hls_clock = dict_default(config, "hls_clock", "5")
        self.axi_bitW = dict_default(config, "axi_bitW", "32")
        self.axi_burstS = dict_default(config, "axi_burstS", "16")
        self.fpga_hz = dict_default(config, "hlx_Mhz", "200")
        self.hlx_script = (
            f"{sc['secda_tflite_path']}/{sc['hlx_scripts']}/{config['hlx_tcl_script']}"
        )

        # misc
        self.board_dir = self.bconfig["board_dir"] + "/bitstreams"
        self.board_script = config["board_script"]
        self.bitstream = (
            config["acc_name"]
            + "_"
            + str(config["acc_version"])
            + "_"
            + str(config["acc_sub_version"])
        )
        self.acc_tag = (
            str(self.acc_name)
            + "_"
            + str(self.acc_version)
            + "_"
            + str(self.acc_sub_version)
            + str(self.acc_test_id)
        )

    def generate_hls_tcl(self, output_dir):
        s = ""
        s += "open_project -reset " + self.acc_tag + "\n"
        s += "set_top " + self.top + "\n"
        for file in os.listdir(self.acc_link_folder):
            if file.endswith(".cc") or file.endswith(".h"):
                s += (
                    "add_files "
                    + "src/"
                    + file
                    + f' -cflags "-D__SYNTHESIS__, -D{self.board}"\n'
                )
        s += 'open_solution "' + self.acc_tag + '"\n'
        s += "set_part " + self.fpga_part + "\n"
        s += "create_clock -period " + self.hls_clock + " -name default\n"
        s += "config_export -format ip_catalog -rtl verilog -taxonomy /s -vendor xilinx\n"
        s += "csynth_design\n"
        s += "export_design -format ip_catalog\n"
        s += "exit\n"
        with open(f"{output_dir}/hls_script.tcl", "w") as f:
            f.write(s)

    def generate_hlx_tcl(self, output_dir):
        hlx_dict = {
            "top": self.top,
            "axi_bitW": self.axi_bitW,
            "axi_burstS": self.axi_burstS,
            "fpga_hz": self.fpga_hz,
        }
        with open((self.hlx_script)) as f:
            tcl_script = str(mt(f.read()).substitute(hlx_dict))
        with open(f"{output_dir}/hlx_script.tcl", "w") as f:
            f.write(tcl_script)

    def create_run_script(self, output_path):
        output_dir = output_path + "/" + self.acc_tag
        run_dict = {
            "acc_link_folder": self.acc_link_folder,
            "acc_tag": self.acc_tag,
            "vp": self.vp,
            "vp_hls": self.vp_hls,
            "bitstream": self.bitstream,
            "board": self.board,
            "board_dir": self.board_dir,
            "board_user": self.bconfig["board_user"],
            "board_hostname": self.bconfig["board_hostname"],
            "board_port": self.bconfig["board_port"],
            "board_script": self.board_script,
            "hlx_version": self.bconfig["hlx_version"],
            "svp_hls": self.svp_hls,
            "svp_hlx": self.svp_hlx,
            "server_user": self.sc["remote_server"]["server_user"],
            "server_hostname": self.sc["remote_server"]["server_hostname"],
            "server_port": self.sc["remote_server"]["server_port"],
            "server_dir": self.sc["remote_server"]["server_dir"],
            "gateway": self.gateway,
            "pb_token": self.sc["push_bullet_token"],
            "stderr": "2>&1",
        }
        with open(hw_gen_tpl) as f:
            run_script = str(mt(f.read()).substitute(run_dict))

        with open(f"{output_dir}/run.sh", "w") as f:
            f.write(run_script)
        os.system(f"chmod +x {output_dir}/run.sh")

    def create_project(self, output_path):
        output_dir = output_path + "/" + self.acc_tag
        output_acc_src_dir = output_dir + "/src"
        os.system(f"mkdir -p {output_dir}")
        os.system(f"mkdir -p {output_acc_src_dir}")
        self.generate_hls_tcl(output_dir)
        self.generate_hlx_tcl(output_dir)
        self.create_run_script(output_path)


def process_hw_config(hw_config_file):
    if hw_config_file.endswith(".json") == False:
        hw_config_file += ".json"

    # Loads the system configuration for SECDA-TFLite
    sc = load_config("../config.json")

    # Loads the hardware configuration
    if not os.path.exists(hw_config_file):
        hw_config_file = find_hw_config(
            f"{sc['secda_tflite_path']}/{sc['hw_configs']}",
            hw_config_file.replace(".json", ""),
        )
    hw_config = load_config(hw_config_file)

    # Creates the necessary directories
    out_dir = sc["out_dir"]
    hw_link_dir = f"{sc['secda_tflite_path']}/{sc['hw_link_dir']}/"
    if not os.path.exists(hw_link_dir + hw_config["acc_link_folder"]):
        os.makedirs(hw_link_dir + hw_config["acc_link_folder"])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Symlinks the accelerator source files to the hardware link directory
    acc_src_dir = (
        f"{sc['secda_tflite_path']}/{sc['path_to_dels']}/{hw_config['acc_src']}/"
    )
    for file in os.listdir(acc_src_dir):
        if file.endswith(".cc") or file.endswith(".h"):
            source = os.path.abspath(acc_src_dir + file)
            target = os.path.abspath(hw_link_dir + hw_config["acc_link_folder"] + "/")
            os.system(f"ln -sf {source} {target}")
    target = os.path.abspath(hw_link_dir + hw_config["acc_link_folder"] + "/")
    sysc_types_path = f"{sc['secda_tools_path']}/secda_integrator/sysc_types.h"
    sysc_hw_utils_path = (
        f"{sc['secda_tools_path']}/secda_integrator/secda_hw_utils.sc.h"
    )
    os.system(f"ln -sf {sysc_types_path} {target}")
    os.system(f"ln -sf {sysc_hw_utils_path} {target}")

    ## Creates the hw_exp project
    acc_proj = hardware_exp(hw_config, sc)
    acc_proj.create_project(out_dir)

    # Prints commands to run the script for the user
    acc_proj_path = os.path.abspath(out_dir + "/" + acc_proj.acc_tag)
    print(f"The project has been created in {acc_proj_path}")
    print(f"To run the project HLS and HLX, run the following commands:")
    print(f"cd {acc_proj_path}")
    print(f"{acc_proj_path}/run.sh")


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: hw_gen.py <config_file>")
        sys.exit(1)

    hw_config_file = args[0]
    if hw_config_file == "ALL":
        for file in os.listdir("./configs/"):
            if file.endswith(".json"):
                process_hw_config(file)
    else:
        process_hw_config(hw_config_file)


if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    main()
