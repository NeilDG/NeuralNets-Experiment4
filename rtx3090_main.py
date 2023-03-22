import os
import multiprocessing
import time


def train_depth():
    os.system("python \"train_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.02\" "
              "--iteration=10")

    os.system("python \"train_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.02\" "
              "--iteration=11")

    # FOR TESTING
    # os.system("python \"train_main.py\" --server_config=3 --img_to_load=-1 "
    #           "--plot_enabled=1 --save_every_iter=50 --network_version=\"depth_v01.13\" "
    #           "--iteration=12")

def test_depth():
    os.system("python \"test_main.py\" --server_config=3 --img_to_load=-1 --plot_enabled=1 --network_version=\"depth_v01.02\" "
              "--iteration=10")

def main():
    # train_depth()
    test_depth()
    # os.system("shutdown /s /t 1")

if __name__ == "__main__":
    main()