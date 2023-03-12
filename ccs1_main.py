import os
import GPUtil

def main():
    GPUtil.showUtilization()
    device_id = GPUtil.getFirstAvailable(maxMemory=0.1, maxLoad=0.1, attempts=50, interval=1800, verbose=True)
    gpu_device = "cuda:" + str(device_id[0])
    print("Available GPU device found: ", gpu_device)

    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.01\" "
             "--iteration=2")

    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.01\" "
             "--iteration=3")

    os.system("python \"train_main.py\" --server_config=1 --img_to_load=-1 --cuda_device=" +gpu_device+ " "
             "--plot_enabled=0 --network_version=\"depth_v01.01\" "
             "--iteration=4")

if __name__ == "__main__":
    main()