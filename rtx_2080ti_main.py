import os

def train_depth():
    os.system("python \"train_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.10\" "
              "--iteration=12")

    os.system("python \"train_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.10\" "
              "--iteration=13")

    os.system("python \"train_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.10\" "
              "--iteration=14")

    os.system("python \"train_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.10\" "
              "--iteration=15")

    os.system("python \"train_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.10\" "
              "--iteration=16")

    os.system("python \"train_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.10\" "
              "--iteration=17")

    os.system("python \"train_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.10\" "
              "--iteration=18")

    os.system("python \"train_main.py\" --server_config=2 --img_to_load=-1 "
              "--plot_enabled=0 --save_every_iter=500 --network_version=\"depth_v01.10\" "
              "--iteration=19")

def test_depth():
    os.system("python \"test_main.py\" --server_config=2 --img_to_load=-1 --plot_enabled=1 --network_version=\"depth_v01.07\" "
              "--iteration=10")

def main():
    train_depth()
    # test_depth()
    os.system("shutdown /s /t 1")

if __name__ == "__main__":
    main()