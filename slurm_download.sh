#!/bin/bash
#SBATCH -J DOWNLOAD
#SBATCH --partition=serial
#SBATCH --qos=84c-1d_serial
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=script_download.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

#About this script:
#Download of dataset
SERVER_CONFIG=7

module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5
source activate NeilGAN_V2

#do fresh install
pip-review --local --auto
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install scikit-learn
pip install scikit-image
pip install visdom
pip install kornia
pip install opencv-python
pip install --upgrade pillow
pip install lpips
pip install gputil
pip install matplotlib
pip install --upgrade --no-cache-dir gdown
pip install PyYAML

if [ $SERVER_CONFIG == 0 ]
then
  srun python "gdown_download.py" --server_config=$SERVER_CONFIG
else
  python "gdown_download.py" --server_config=$SERVER_CONFIG
fi

DATASET_NAME="v01_fcity"

if [ $SERVER_CONFIG == 0 ]
then
  OUTPUT_DIR="/scratch1/scratch2/neil.delgallego/SynthV3_Raw/"
elif [ $SERVER_CONFIG == 4 ]
then
  OUTPUT_DIR="D:/Datasets/SynthV3_Raw/"
elif [ $SERVER_CONFIG == 7 ]
then
  OUTPUT_DIR="/SynthV3_Raw/"
else
  OUTPUT_DIR="/home/jupyter-neil.delgallego/SynthV3_Raw/"
fi

echo "$OUTPUT_DIR/$DATASET_NAME.zip"

zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"
#mv "$OUTPUT_DIR/$DATASET_NAME+fixed" "$OUTPUT_DIR/$DATASET_NAME"
#rm -rf "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"

if [ $SERVER_CONFIG == 4 ]
then
  python "rl208_main.py"
fi