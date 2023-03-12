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
SERVER_CONFIG=1 #0 = COARE, 1 = CCS Cloud

module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5
source activate NeilGAN_V2

pip install --upgrade --no-cache-dir gdown

if [ $SERVER_CONFIG == 0 ]
then
  srun python "gdown_download.py" --server_config=$SERVER_CONFIG
else
  python "gdown_download.py" --server_config=$SERVER_CONFIG
fi

DATASET_NAME="v50_places"

if [ $SERVER_CONFIG == 0 ]
then
  OUTPUT_DIR="/scratch1/scratch2/neil.delgallego/SynthV3_Raw/"
else
  OUTPUT_DIR="/home/jupyter-neil.delgallego/SynthV3_Raw/"
fi

echo "$OUTPUT_DIR/$DATASET_NAME.zip"

zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"
mv "$OUTPUT_DIR/$DATASET_NAME+fixed" "$OUTPUT_DIR/$DATASET_NAME"
rm -rf "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"

#if [ $SERVER_CONFIG == 1 ]
#then
#  python "ccs1_main.py"
#fi