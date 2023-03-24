import os
import multiprocessing
import time


def train_depth():
    # os.system("python \"train_main.py\" --server_config=3 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.02\" "
    #           "--iteration=10")

    os.system("python \"train_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.02\" "
              "--iteration=11")

    # FOR TESTING
    # os.system("python \"train_main.py\" --server_config=3 --img_to_load=-1 "
    #           "--plot_enabled=1 --save_every_iter=50 --network_version=\"depth_v01.13\" "
    #           "--iteration=12")

def test_depth():
    # os.system("python \"test_main.py\" --server_config=3 --img_to_load=-1 --plot_enabled=0 --network_version=\"depth_v01.10\" "
    #           "--iteration=12")
    #
    # os.system("python \"test_main.py\" --server_config=3 --img_to_load=-1 --plot_enabled=0 --network_version=\"depth_v01.10\" "
    #           "--iteration=13")
    #
    # os.system("python \"test_main.py\" --server_config=3 --img_to_load=-1 --plot_enabled=0 --network_version=\"depth_v01.10\" "
    #           "--iteration=14")
    #
    # os.system("python \"test_main.py\" --server_config=3 --img_to_load=-1 --plot_enabled=0 --network_version=\"depth_v01.10\" "
    #           "--iteration=15")
    #
    # os.system("python \"test_main.py\" --server_config=3 --img_to_load=-1 --plot_enabled=0 --network_version=\"depth_v01.10\" "
    #           "--iteration=16")
    #
    # os.system("python \"test_main.py\" --server_config=3 --img_to_load=-1 --plot_enabled=0 --network_version=\"depth_v01.10\" "
    #           "--iteration=17")
    #
    # os.system("python \"test_main.py\" --server_config=3 --img_to_load=-1 --plot_enabled=0 --network_version=\"depth_v01.10\" "
    #           "--iteration=18")

    os.system("python \"test_main.py\" --server_config=3 --img_to_load=-1 --plot_enabled=1 --network_version=\"depth_v01.10\" "
              "--iteration=19")

def train_img2img():
    os.system("python \"train_img2img_main.py\" --server_config=3 --img_to_load=1000 "
              "--plot_enabled=0 --save_every_iter=100 --network_version=\"synth2real_v01.00\" "
              "--iteration=1")

def test_img2img():
    os.system("python \"test_img2img_main.py\" --server_config=3 --img_to_load=1000 "
              "--plot_enabled=1 --network_version=\"synth2real_v01.00\" "
              "--iteration=1")

def main():
    # train_depth()
    test_depth()

    # train_img2img()
    # test_img2img()
    # os.system("shutdown /s /t 1")

if __name__ == "__main__":
    main()