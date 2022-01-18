<!---
This is the code used in the paper: [Property-driven Training: All You (N)Ever Wanted to Know About](https://arxiv.org/abs/2104.01396)
--->
This is the code used in the paper: Neural Network Robustness as a Mathematical Property: A Principled Case Study

TRAINING WITH CONSTRAINTS
========

Structure
------------
```
.
├── src
│   ├── dl2                                  - DL2 Library
│   ├── createBaselineDatasets.py            - script for creating the baseline datasets
│   ├── createAugmentedDatasets.py           - script for creating the Random Uniform augmented datasets
│   ├── createFGSMAugmentedDatasets.py       - script for creating the FGSM augmented datasets
│   ├── mainDL2.py                           - script for DL2 loss training
│   ├── mainPGD.py                           - script for Adversarial training
│   ├── testConstraintSecurity.py            - script for testing Constraint Security
├── createFolders.sh                     - script for creating the required folders for datasets and models
└── requirements.txt                     - pip requirements

```

Installation
------------
Clone this repository:
```
git clone https://github.com/aisec-private/training-with-constraints.git
cd training-with-constraints
```
Install the dependencies:
```
pip install -r requirements.txt
```
To properly save the datasets and models create the following structure of folders using ```createFolders.sh```:
```
.
├── datasets
│   ├── fashion_mnist
│   │   ├── baseline
│   │   ├── augmented
│   │   └── augmented_FGSM
│   └── gtsrb
│       ├── baseline
│       ├── augmented
│       └── augmented_FGSM
└── models
    ├── fashion_mnist
    │   ├── baseline
    │   ├── augmented
    │   ├── augmented_FGSM
    │   └── dl2
    └── gtsrb
        ├── baseline
        ├── augmented
        ├── augmented_FGSM
        └── dl2
```

Reproducing Experiments and Results
-------------
To generate the datasets, run the commands:
```
python createBaselineDatasets.py

python createAugmentedDatasets.py --dataset fashion_mnist
python createAugmentedDatasets.py --dataset gtsrb

python createFGSMAugmentedDatasets.py
```
To reproduce the experiments in the paper run the following commands:
```
python mainDL2.py --dataset fashion_mnist --dtype baseline --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainDL2.py --dataset fashion_mnist --dtype augmented --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainDL2.py --dataset fashion_mnist --dtype augmented_FGSM --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainPGD.py --dataset fashion_mnist --dtype baseline --alfa 0 --beta 1
python mainDL2.py --dataset fashion_mnist --dtype baseline --dl2-weight 0.2 --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainDL2.py --dataset fashion_mnist --dtype baseline --dl2-weight 0.2 --constraint "RobustnessG(eps=0.1, delta=0.52)"
python mainDL2.py --dataset fashion_mnist --dtype baseline --dl2-weight 0.2 --constraint "LipschitzG(eps=0.1, L=10)"
python mainDL2.py --dataset fashion_mnist --dtype baseline --dl2-weight 0.2 --constraint "FGSM(eps=0.1, delta=10)"

python mainDL2.py --dataset gtsrb --dtype baseline --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainDL2.py --dataset gtsrb --dtype augmented --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainDL2.py --dataset gtsrb --dtype augmented_FGSM --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainPGD.py --dataset gtsrb --dtype baseline --alfa 0 --beta 1
python mainDL2.py --dataset gtsrb --dtype baseline --dl2-weight 0.2 --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainDL2.py --dataset gtsrb --dtype baseline --dl2-weight 0.2 --constraint "RobustnessG(eps=0.1, delta=0.52)"
python mainDL2.py --dataset gtsrb --dtype baseline --dl2-weight 0.2 --constraint "LipschitzG(eps=0.1, L=10)"
python mainDL2.py --dataset gtsrb --dtype baseline --dl2-weight 0.2 --constraint "FGSM(eps=0.1, delta=10)"
```
To test Constraint Security of the models obtained by the experiments above, run the command:
```
python testConstraintSecurity.py
```
