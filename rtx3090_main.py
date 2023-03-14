import os

def train_depth():
    os.system("python \"train_main.py\" --server_config=3 --img_to_load=-1 "
             "--plot_enabled=0 --network_version=\"depth_v01.01\" "
             "--iteration=1")

def test_depth():
    os.system("python \"test_main.py\" --server_config=4 --img_to_load=-1 --network_version=\"v01.01\" "
              "--iteration=3")
def main():
    train_depth()
    # test_depth()
    # os.system("shutdown /s /t 1")

if __name__ == "__main__":
    main()