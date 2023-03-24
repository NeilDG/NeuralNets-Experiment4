import os
import multiprocessing
import time


def train_depth():
    os.system("python3 \"train_main.py\" --server_config=6 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.12\" "
              "--iteration=1")

def test_depth():
    os.system("python3 \"test_main.py\" --server_config=6 --img_to_load=-1 --plot_enabled=1 --network_version=\"depth_v01.04\" "
              "--iteration=12")

def main():
    train_depth()
    # test_depth()
    # os.system("shutdown /s /t 1")

if __name__ == "__main__":
    main()