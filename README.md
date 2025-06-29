# Generalized motion prediction code
Code for the paper "Autoregressive Flow Matching for Motion Prediction"

## Installation
Note by default everything is installed as editable but this isn't necessary if you don't intend to modify anything
### Base installation (training and inference of motion prediction models)
```
git clone https://github.com/Johnathan-Xie/arfm-motion-prediction
conda create --name motion_prediction_env python=3.10
conda activate motion_prediction_env
cd datasets
pip install -e .
```

```
cd ../tapnet
pip install -e .
mkdir checkpoints
cd checkpoints
wget https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
wget https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt
```

```
cd ../motion_prediction
pip install -r requirements.txt
pip install -e .
cd ../sam2

pip install -e .
```

### Robotics application installation (for CALVIN experiments), requires base installation
Go into the robotics directory and follow the directions in that README.md
## Code layout
Below we describe the layout of the code. If you just want the 
commands to run experiments then you may skip this section. However, if you plan on modifying
the code or using it beyond just running experiments we highly recommend that you read through
this and we try to keep it brief.

* datasets/: We write a custom implementation of datasets that supports video files (at the start of
    this project this did not exist) and has better storage efficiency for numpy arrays because
    we store tracks at fp16 precision and huggingface converts to python lists which more than
    quadruple storage. Track storage is actually quite expensive as we store the raw values
    and not a compressed format. These modifications are relatively easy to port and are located
    at datasets/src/datasets/features/video.py (video) and datasets/src/datasets/features/array.py
    (numpy arrays)

* motion_prediction/: This is our main code for the actual motion predictor and query predictor models. We provide
    code for training, inference, and evaluation.

    * motion_prediction/motion_predictor/: Modeling code for motion predictor.
    * motion_prediction/query_predictor/: Modeling code for query predictor.
    * motion_prediction/spacetime_transformer: Feature fusion transformer architecture used by feature fusion module
    * motion_prediction/prediction_head/: Flow matching prediction head architecture used by flow matching predictor

    * evaluation/: Evaluation utils
    * inference_pipelines/: Encapsulates online point tracker + motion predictor logic to perform updating horizon predictions
    * modeling_track_encoder/: Track encoder architeture used for encoding future tracks to feature vector for downstream tasks.
    * run_training_motion_predictor.py: Training script for motion prediction model
    * run_evaluation_motion_predictor.py: Evaluation script for motion prediction model


## Experiments
In general for all experiments, the datasets have already been preprocessed and prepared for you to ease reproduction
of results (and to decrease the chance you mess up the preprocessing). If you are curious how the preprocessing was done
please reach out to me, but in general the processes are pretty straight forward.

Below we describe the experiments for training and evaluating a motion prediction model while the application experiments are in the robotics/ directory.

### Training
Training of the zero-shot general model can be done with the following command. Further details and options are present
in the training script itself.
```
bash run_motion_predictor_training.sh
```

### Evaluation
Evaluation of a model can be done with the evaluation script. Options for selecting an evaluation dataset and model to evaluate are available in the evaluation script
```
bash run_motion_predictor_evaluation.sh
```
## Acknowledgement
transformers
datasets
accelerate

## Correspondence
If you have questions about the paper, code, or are opening an issue that requires immediate attention, please also email me at johnathanxie123@gmail.com for
a faster response.

## Citation
