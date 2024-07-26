# Bone_Tumor_Xray_Preprocessing

An optimised preprocessing and data augmentation pipeline targeted for bone tumor X-ray images for the downstream task of a three-class classification (benign, intermediate, malignant). 

The project contains additional implementation of preprocessing techniques that have potential to contribute to the optimal preprocessing depending on the dataset.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```
git clone https://github.com/aleksandraiva/Bone_Tumor_Xray_Preprocessing.git
cd Bone_Tumor_Xray_Preprocessing
```

### Step 2: Create and Activate the Conda Environment
Use the environment.yml file to create and activate the conda environment:

```
conda env create -f environment.yml
conda activate bone_tumor_cl
```

## Usage

Define data directories and metadata files in the k_fold.py file. 

For further experiments, you can change the train_transforms and test_transforms by adding or removing preprocessing and/or augmentation steps. 

Adapt hyperparameters if needed, as the model showed dependency on hyperparameter combinations. The provided hyperparameters showed optimal performance on the dataset and the defined downstream task. 

By running the k_fold.py script in the 'train' folder all outputs will be saved in the same 'train' folder. Run the script by using:

```
cd Bone_Tumor_Xray_Preprocessing/train
nohup python k_fold.py > results.log 2>&1 &
```







