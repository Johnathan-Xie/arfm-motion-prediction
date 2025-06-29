# Robotics Application

## Installation

Setup conda environment and install the CALVIN benchmark on top of existing motion prediction env
```
conda activate motion_prediction_env
cd robotics/
pip install setuptools==57.5.0
cd calvin
cd calvin_env
cd ../calvin_models
sed -i 's/pytorch-lightning==1.8.6/pytorch-lightning/g' requirements.txt
sed -i 's/torch==1.13.1/torch/g' requirements.txt
cd ..
sh ./install.sh
cd ..
```
Install this repository:
```
pip install -r requirements.txt
apt-get install -y libegl1-mesa libegl1
apt-get install -y libgl1
apt-get install -y libosmesa6-dev
apt-get install -y patchelf
```
Get weights
```
mkdir checkpoints
cd checkpoints
git clone https://huggingface.co/jxie/calvin-abcd_d-10_percent-robotics_model
git clone https://huggingface.co/jxie/calvin-abc_d-robotics_model
git clone https://huggingface.co/jxie/calvin-mae_pretrain_vit_base
```
Get validation datasets
```
cd ..
mkdir datasets
cd datasets
git clone https://huggingface.co/datasets/jxie/calvin-lmdb-abcd_d-10_percent
git clone https://huggingface.co/datasets/jxie/calvin-lmdb-abc_d
```
## Experiments
Use the script run_evaluation.sh to evaluate models

## Acknowledgement
This code is based on [GR1-Training](https://github.com/EDiRobotics/GR1-Training)