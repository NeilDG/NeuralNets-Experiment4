import os

def main():
    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 "
             "--plot_enabled=1 --network_version=\"depth_v01.03\" "
             "--iteration=3")

if __name__ == "__main__":
    main()