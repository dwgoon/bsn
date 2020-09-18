import yaml
import platform
import os
from os.path import join as pjoin
from os.path import dirname, basename, abspath
import tempfile
from datetime import datetime
import copy
import argparse

fstr_dname_ckpt = "ckpt_%s_%s_%s_bps-%s"

def set_model_specific_options(cfg_tmp, cfg, cfg_job):
    if cfg_job["MODEL"].startswith("srnet"):
        cfg_tmp["KERNEL_SIZE"] = cfg_job["KERNEL_SIZE"]
            
def print_model_specific_options(cfg_tmp):
    if cfg_job["MODEL"].startswith("srnet"):
        print("- KERNEL SIZE:", cfg_tmp["KERNEL_SIZE"])

def find_dpath_ckpt(droot_ckpt, dname_ckpt):
    """Find the directory of a given dname_ckpt in droot_ckpt.
    """
    for entity in os.listdir(droot_ckpt):
        dpath = pjoin(droot_ckpt, entity)
        if os.path.isdir(dpath) and dname_ckpt in entity:
            return pjoin(droot_ckpt, entity)            
    
    return None
        
def find_test_file(dpath_test, fname_test):
    """Find the directory of a given fname_test in dpath_test.
    """
    for entity in os.listdir(dpath_test):
        fpath = pjoin(dpath_test, entity)
        if os.path.isfile(fpath) and fname_test in entity:
            return pjoin(dpath_test, entity)            
    
    return None

def launch_process(cfg_tmp, mode):
    mode = mode.lower()
    if mode not in ["train","test"]:
        raise ValueError("mode should be 'train' or 'test'.")
    
    fp_cfg = tempfile.NamedTemporaryFile(suffix=".yml")        
    with open(fp_cfg.name, "wt") as fout:
        yaml.dump(cfg_tmp, fout)
        
    os.system("python -u {}.py --config {}".format(mode, fp_cfg.name))

def curriculum_learning(cfg, cfg_job):
    
    global fstr_dname_ckpt    
    fstr_dname_cover = "cover_audio_{DS}_wav_1000msec"
    fstr_dname_stego = "stego_audio_{DS}_{SG}"\
                       "_wav-16bits_1000msec_16000hz_1ch"\
                       "_bitrate-{BPS}_randkey"
    
    
    hostname = platform.node().upper()
    droot_dataset = cfg["SYSTEM_SPECIFIC"][hostname]["DROOT_DATASET"]
    droot_ckpt = abspath(cfg["SYSTEM_SPECIFIC"][hostname]["DROOT_CKPT"])
    os.makedirs(droot_ckpt, exist_ok=True)

    ds = cfg_job["DATASET"].lower()
    model = cfg_job["MODEL"].lower()
    sg = cfg_job["STEGANOGRAPHY"].lower()
    
    
    if len(cfg_job["TRAIN"]["BPS"]) != len(cfg_job["TRAIN"]["EPOCHS"]):
        raise ValueError("Num. items in BPS and Num. items in EPOCHS "\
                         "should be equal for curriculum learning.")
    
    for i, bps in enumerate(cfg_job["TRAIN"]["BPS"]):
        n_epochs = cfg_job["TRAIN"]["EPOCHS"][i]
        cfg_tmp = copy.deepcopy(cfg)                    
        cfg_tmp["MODEL"] = model
        cfg_tmp["DATASET"] = ds
        cfg_tmp["STEGANOGRAPHY"] = sg
        cfg_tmp["BPS"] = bps
        cfg_tmp["TRAIN"]["N_EPOCHS"] = n_epochs
        set_model_specific_options(cfg_tmp, cfg, cfg_job)
        
        # Set directories of cover and stego
        dname_cover = fstr_dname_cover.format(DS=ds)
        dpath_cover = pjoin(droot_dataset, ds, dname_cover)
        
        dname_stego = fstr_dname_stego.format(DS=ds, SG=sg, BPS=bps)
        dpath_stego = pjoin(droot_dataset, ds, dname_stego)        
                        
        cfg_tmp["SYSTEM_SPECIFIC"][hostname]["DPATH_COVER"] = dpath_cover
        cfg_tmp["SYSTEM_SPECIFIC"][hostname]["DPATH_STEGO"] = dpath_stego
        
        dpath_load_ckpt = ""
        dpath_save_ckpt = ""
        
        # Set dpath_save_ckpt for training
        dname_save_ckpt = fstr_dname_ckpt%(model, ds, sg, bps)
        dpath_save_ckpt = find_dpath_ckpt(droot_ckpt, dname_save_ckpt)
        train_already_done = False
        if dpath_save_ckpt:            
            train_already_done = True
        else:  # Perform training        
            now = datetime.now().strftime("%y%m%d")
            dname_save_ckpt = dname_save_ckpt + "_" + now
            dpath_save_ckpt = pjoin(droot_ckpt, dname_save_ckpt)
            cfg_tmp["SYSTEM_SPECIFIC"][hostname]["DPATH_SAVE_CKPT"]\
                                                              = dpath_save_ckpt
            
            # Set dpath_load_ckpt for training
            CL_INIT_BPS = cfg_job["TRAIN"]["CL_INIT_BPS"]
            if bps == CL_INIT_BPS:
                dpath_load_ckpt = ""
            else:
                if i != 0:
                    bps_prev = cfg_job["TRAIN"]["BPS"][i-1] 
                else:
                    bps_prev = CL_INIT_BPS
                    
                dname_load_ckpt = fstr_dname_ckpt%(model, ds, sg, bps_prev)
                dpath_load_ckpt = find_dpath_ckpt(droot_ckpt, dname_load_ckpt)
                if not dpath_load_ckpt:
                    err_msg = "Cannot find the directory to load!"
                    raise NotADirectoryError(err_msg)
            
                cfg_tmp["SYSTEM_SPECIFIC"][hostname]["DPATH_LOAD_CKPT"]\
                                                              = dpath_load_ckpt
            # end of if-else
        # end of if-else
        
        # Set dpath_load_ckpt for testing
        """[!] Note that we use DPATH_SAVE_CKPT of training 
               for DPATH_LOAD_CKPT of testing.
        """
            
        # Train
        print("[CL JOB #{} train started...]".format(i+1))
        print("- MODEL:", model)
        print_model_specific_options(cfg_tmp)

        print("- DATASET:", ds)
        print("- STEGANOGRAPHY:", sg)
        print("- BPS:", bps)       
        
        if train_already_done:
            print("[!] SKIP TRAINING")
            print("[!] %s has been already done... "%(dname_save_ckpt))
        else:
            print("- N_EPOCHS:", n_epochs)
            print("- DPATH_COVER:", dpath_cover)
            print("- DPATH_STEGO:", dpath_stego)
            print("- DPATH_SAVE_CKPT:", dpath_save_ckpt)
            print("- DPATH_LOAD_CKPT:", dpath_load_ckpt)        
            launch_process(cfg_tmp, mode="train")
        # end of if-else
        print("[CL JOB #{} train finished...]".format(i+1))
        print()

        # Test
        dpath_load_ckpt = dpath_save_ckpt
        print("[CL JOB #{} test started...]".format(i+1))
        print("- N_REPEATS:", cfg_job["TEST"]["N_REPEATS"])
        print("- DPATH_LOAD_CKPT:", dpath_load_ckpt)
        
        cfg_tmp["SYSTEM_SPECIFIC"][hostname]["DPATH_LOAD_CKPT"]\
                                                              = dpath_load_ckpt        
        cfg_tmp["TEST"]["N_REPEATS"] = cfg_job["TEST"]["N_REPEATS"]


        dname_ckpt = os.path.basename(dpath_load_ckpt)
        dname_ckpt = '_'.join(dname_ckpt.split('_')[:-1])
        dpath_test = cfg["SYSTEM_SPECIFIC"][hostname]["DPATH_TEST"]
        os.makedirs(dpath_test, exist_ok=True)
        if find_test_file(dpath_test, "test_{}".format(dname_ckpt)):
            print("[!] SKIP TEST")
            print("[!] %s has been already done... "%(dname_ckpt))
        else:
            launch_process(cfg_tmp, mode="test")
        
        print("[CL JOB #{} test finished...]".format(i+1))
        print()
        
        
    
    
def individual_learning(cfg, cfg_job):
    global fstr_dname_ckpt    
    fstr_dname_cover = "cover_audio_{DS}_wav_1000msec"
    fstr_dname_stego = "stego_audio_{DS}_{SG}"\
                       "_wav-16bits_1000msec_16000hz_1ch"\
                       "_bitrate-{BPS}_randkey"
    
    
    hostname = platform.node().upper()
    if "FORENSIC" in hostname:
        hostname = "FORENSIC"
        
    droot_dataset = cfg["SYSTEM_SPECIFIC"][hostname]["DROOT_DATASET"]
    droot_ckpt = abspath(cfg["SYSTEM_SPECIFIC"][hostname]["DROOT_CKPT"])
    os.makedirs(droot_ckpt, exist_ok=True)
    
    ds = cfg_job["DATASET"].lower()
    model = cfg_job["MODEL"].lower()
    sg = cfg_job["STEGANOGRAPHY"].lower()

    bps = cfg_job["TRAIN"]["BPS"]
    n_epochs = cfg_job["TRAIN"]["N_EPOCHS"]
    
    cfg_tmp = copy.deepcopy(cfg)                    
    cfg_tmp["MODEL"] = model
    cfg_tmp["DATASET"] = ds
    cfg_tmp["STEGANOGRAPHY"] = sg
    cfg_tmp["BPS"] = bps
    cfg_tmp["TRAIN"]["N_EPOCHS"] = n_epochs
    set_model_specific_options(cfg_tmp, cfg, cfg_job)
    
    # Set directories of cover and stego
    dname_cover = fstr_dname_cover.format(DS=ds)
    dpath_cover = pjoin(droot_dataset, ds, dname_cover)
    
    dname_stego = fstr_dname_stego.format(DS=ds, SG=sg, BPS=bps)
    dpath_stego = pjoin(droot_dataset, ds, dname_stego)        
                    
    cfg_tmp["SYSTEM_SPECIFIC"][hostname]["DPATH_COVER"] = dpath_cover
    cfg_tmp["SYSTEM_SPECIFIC"][hostname]["DPATH_STEGO"] = dpath_stego
    
    dpath_load_ckpt = ""
    dpath_save_ckpt = ""
    
    # Set dpath_save_ckpt for training
    dname_save_ckpt = fstr_dname_ckpt%(model, ds, sg, bps)
    dpath_save_ckpt = find_dpath_ckpt(droot_ckpt, dname_save_ckpt)
    train_already_done = False
    if dpath_save_ckpt:            
        train_already_done = True
    else:  # Perform training        
        now = datetime.now().strftime("%y%m%d")
        dname_save_ckpt = dname_save_ckpt + "_" + now
        dpath_save_ckpt = pjoin(droot_ckpt, dname_save_ckpt)
        cfg_tmp["SYSTEM_SPECIFIC"][hostname]["DPATH_SAVE_CKPT"]\
                                                          = dpath_save_ckpt
    
    # Set dpath_load_ckpt for testing
    """[!] Note that we use DPATH_SAVE_CKPT of training 
           for DPATH_LOAD_CKPT of testing.
    """
    
        
    # Train
    print("[IL JOB for {}] train started...".format(model.upper()))
    print("- MODEL:", model)
    print_model_specific_options(cfg_tmp)

    print("- DATASET:", ds)
    print("- STEGANOGRAPHY:", sg)
    print("- BPS:", bps)
    
    if train_already_done:
        print("[!] SKIP TRAINING")
        print("[!] %s has been already done... "%(dname_save_ckpt))
    else:
        print("- N_EPOCHS:", n_epochs)
        print("- DPATH_COVER:", dpath_cover)
        print("- DPATH_STEGO:", dpath_stego)
        print("- DPATH_SAVE_CKPT:", dpath_save_ckpt)
        print("- DPATH_LOAD_CKPT:", dpath_load_ckpt)        
        launch_process(cfg_tmp, mode="train")
    # end of if-else
    print("[IL JOB for {}] train finished...".format(model.upper()))
    print()

    # Test
    dpath_load_ckpt = dpath_save_ckpt
    print("[IL JOB for {}] test started...".format(model.upper()))
    print("- N_REPEATS:", cfg_job["TEST"]["N_REPEATS"])
    print("- DPATH_LOAD_CKPT:", dpath_load_ckpt)
    
    cfg_tmp["SYSTEM_SPECIFIC"][hostname]["DPATH_LOAD_CKPT"]\
                                                          = dpath_load_ckpt        
    cfg_tmp["TEST"]["N_REPEATS"] = cfg_job["TEST"]["N_REPEATS"]


    dname_ckpt = os.path.basename(dpath_load_ckpt)
    dname_ckpt = '_'.join(dname_ckpt.split('_')[:-1])
    dpath_test = cfg["SYSTEM_SPECIFIC"][hostname]["DPATH_TEST"]
    os.makedirs(dpath_test, exist_ok=True)
    if find_test_file(dpath_test, "test_{}".format(dname_ckpt)):
        print("[!] SKIP TEST")
        print("[!] %s has been already done... "%(dname_ckpt))
    else:
        launch_process(cfg_tmp, mode="test")
    
    print("[IL JOB for {}] test finished...".format(model.upper()))
    print()
# end of def
        
        

parser = argparse.ArgumentParser(description='Parse the configruation for automating train and test.')
parser.add_argument('--config',
                    dest='config',
                    action='store',
                    type=str,
                    help="Designate the file path of " \
                         "the configuration file in YAML")


args = parser.parse_args()
fpath_cfg = args.config
print("Configuration file:", fpath_cfg)

with open(fpath_cfg, "rt") as fin:
    cfg = yaml.safe_load(fin)
    

for i, cfg_job in enumerate(cfg["JOBS"]):  # Configuration of a single job
    if cfg_job["TRAIN"]["LEARNING_TYPE"] == "CL":
        print("[CURRICULUM LEARNING #%d]"%(i+1))        
        curriculum_learning(cfg, cfg_job)
    elif cfg_job["TRAIN"]["LEARNING_TYPE"] == "IL":
        print("[INDIVIDUAL LEARNING #%d]"%(i+1))
        individual_learning(cfg, cfg_job)
