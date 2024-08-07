import sys
sys.dont_write_bytecode = True
import json

from utils import *
from gen_config_arr import *
from gen_bins import *

# parse arguments if any
arglen = len(sys.argv)
if arglen > 1:
    aec_path = sys.argv[1] ## apps_evaluation_config path
    gen_bin = sys.argv[2]
    
sc = load_config("../../config.json") ## system config
aec = load_config(aec_path) ## apps evaluation config

board_user = sc["board_user"]
data_dir_host = sc["model_n_data_dir"]
data_dir_fpgaBoard = f"/home/{board_user}/Workspace/secda_apps_evaluation_suite/"+data_dir_host.split("/")[-1]
bitstream_dir = bitstream_dir = f"/home/{board_user}/Workspace/secda_benchmark_suite/bitstreams"
bin_dir = f"/home/{board_user}/Workspace/secda_benchmark_suite/bins"

# print("board_user: ", board_user)
# print("data_dir_host: ", data_dir_host)
# print("data_dir_fpgaBoard: ", data_dir_fpgaBoard)

def analyze_eval_apps_config(aec):
    bm_apps_config = []
    indiff_apps_config = []
    em_apps_config = []
    ema_apps_config = []
    iic_apps_config = []   
    hardware = list(aec[1]["hardware"].values())
    ## find apps configuration to take for the defined apps
    for app_name in aec[0]["apps"].values():
        if app_name == "benchmark_model":
            bm_apps_config = aec[2]["bm_apps_configuration"].values()
            bm_apps_config = list(bm_apps_config)
            for i in range(len(bm_apps_config[0])):
                #serach for data_dir_host in bm_apps_config[0][i]
                #replace data_dir_host with data_dir_fpgaBoard
                ## changing the model directory host to FPGA
                if data_dir_host in bm_apps_config[0][i]:
                    bm_apps_config[0][i] = bm_apps_config[0][i].replace(data_dir_host, data_dir_fpgaBoard)
                    
            bm_apps_config.append(hardware)
            
        if app_name == "inference_diff":
            indiff_apps_config = aec[2]["indiff_apps_configuration"].values()
            indiff_apps_config = list(indiff_apps_config)
            for i in range(len(indiff_apps_config[0])):
                #serach for data_dir_host in indiff_apps_config[0][i]
                #replace data_dir_host with data_dir_fpgaBoard
                ## changing the model directory host to FPGA
                if data_dir_host in indiff_apps_config[0][i]:
                    indiff_apps_config[0][i] = indiff_apps_config[0][i].replace(data_dir_host, data_dir_fpgaBoard)
            
            indiff_apps_config.append(hardware)
        
        if app_name == "eval_model":
            em_apps_config = aec[2]["em_apps_configuration"].values()
            em_apps_config = list(em_apps_config)
            for i in range(len(em_apps_config[0])):
                #serach for data_dir_host in em_apps_config[0][i]
                #replace data_dir_host with data_dir_fpgaBoard
                ## changing the model directory host to FPGA
                if data_dir_host in em_apps_config[0][i]:
                    em_apps_config[0][i] = em_apps_config[0][i].replace(data_dir_host, data_dir_fpgaBoard)
                
                ## changing the image_name directory host to FPGA
                if data_dir_host in em_apps_config[1][i]:
                    em_apps_config[1][i] = em_apps_config[1][i].replace(data_dir_host, data_dir_fpgaBoard)
            em_apps_config.append(hardware)
            
        if app_name == "eval_model_accuracy":
            ema_apps_config = aec[2]["ema_apps_configuration"].values()
            ema_apps_config = list(ema_apps_config)
            for i in range(len(ema_apps_config[0])):
                #serach for data_dir_host in ema_apps_config[0][i]
                #replace data_dir_host with data_dir_fpgaBoard
                ## changing the model directory host to FPGA
                if data_dir_host in ema_apps_config[0][i]:
                    ema_apps_config[0][i] = ema_apps_config[0][i].replace(data_dir_host, data_dir_fpgaBoard)
            
            ema_apps_config.append(hardware)
        
        if app_name == "imagenet_image_classification":
            iic_apps_config = aec[2]["iic_apps_configuration"].values()
            iic_apps_config = list(iic_apps_config)
            for i in range(len(iic_apps_config[0])):
                #serach for data_dir_host in iic_apps_config[0][i]
                #replace data_dir_host with data_dir_fpgaBoard
                ## changing the model directory host to FPGA
                if data_dir_host in iic_apps_config[0][i]:
                    iic_apps_config[0][i] = iic_apps_config[0][i].replace(data_dir_host, data_dir_fpgaBoard)
            
            iic_apps_config.append(hardware)
        
    return bm_apps_config, indiff_apps_config, em_apps_config, ema_apps_config,\
        iic_apps_config, hardware
        
if __name__ == "__main__":
    bm_apps_config, indiff_apps_config, em_apps_config, ema_apps_config, \
        iic_apps_config, hardware = analyze_eval_apps_config(aec)
    
    # remove the files from sc["out_dir"]
    os.system(f"rm -rf {sc['out_dir']}/*")
    
    #create supported tools dictionary
    supported_tools = {}
    
    #rename the model names in the configuration
    if bm_apps_config:
        bm_apps_config.append(data_dir_fpgaBoard)
        bm_apps_config.append(bitstream_dir)
        bm_apps_config.append(bin_dir)
        bm_apps_config.append(board_user)
        print("bm_apps_config: ", bm_apps_config)
        gen_config_arr_imp_bm(sc, bm_apps_config)
        # add a key and value to the supported tools dictionary
        supported_tools["benchmark_model"] = "bm"

    if indiff_apps_config:
        indiff_apps_config.append(data_dir_fpgaBoard)
        indiff_apps_config.append(bitstream_dir)
        indiff_apps_config.append(bin_dir)
        indiff_apps_config.append(board_user)
        print("indiff_apps_config: ", indiff_apps_config)
        gen_config_arr_imp_indiff(sc, indiff_apps_config)
        # add a key and value to the supported tools dictionary
        supported_tools["inference_diff"] = "id"
        
    if em_apps_config:
        em_apps_config.append(data_dir_fpgaBoard)
        em_apps_config.append(bitstream_dir)
        em_apps_config.append(bin_dir)
        em_apps_config.append(board_user)
        print("em_apps_config: ", em_apps_config)
        # print("len(em_apps_config): ", len(em_apps_config))
        gen_config_arr_imp_em(sc, em_apps_config)
        # add a key and value to the supported tools dictionary
        supported_tools["eval_model"] = "em"      
    
    if ema_apps_config:
        ema_apps_config.append(data_dir_fpgaBoard)
        ema_apps_config.append(bitstream_dir)
        ema_apps_config.append(bin_dir)
        ema_apps_config.append(board_user)
        print("ema_apps_config: ", ema_apps_config)
        # print("len(ema_apps_config): ", len(ema_apps_config))
        gen_config_arr_imp_ema(sc, ema_apps_config)
        # add a key and value to the supported tools dictionary
        supported_tools["eval_model_accuracy"] = "ema"
        
    if iic_apps_config:
        iic_apps_config.append(data_dir_fpgaBoard)
        iic_apps_config.append(bitstream_dir)
        iic_apps_config.append(bin_dir)
        iic_apps_config.append(board_user)
        print("iic_apps_config: ", iic_apps_config)
        gen_config_arr_imp_iic(sc, iic_apps_config)
        # add a key and value to the supported tools dictionary
        supported_tools["imagenet_image_classification"] = "iic"
        
    ### generating binary files
    if gen_bin:
        print("Generating bins")
        gen_bins(sc, hardware, supported_tools)