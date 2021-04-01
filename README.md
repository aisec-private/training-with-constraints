TRAINING WITH CONSTRAINTS
========

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
To properly save the datasets and models create the following structure of folders inside the main folder:
```
datasets
      | - fashion_mnist
            | - baseline
            | - augmented
            | - augmented_FGSM
      | - gtsrb
            | - baseline
            | - augmented
            | - augmented_FGSM
models
      | - fashion_mnist
            | - baseline
            | - augmented
            | - augmented_FGSM
            | - dl2
      | - gtsrb
            | - baseline
            | - augmented
            | - augmented_FGSM
            | - dl2
```

Usage
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
python mainFGSM.py --dataset fashion_mnist --dtype baseline --eps 0.1
python mainPGD.py --dataset fashion_mnist --dtype baseline --alfa 0 --beta 1
python mainDL2.py --dataset fashion_mnist --dtype baseline --dl2-weight 0.2 --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainDL2.py --dataset fashion_mnist --dtype baseline --dl2-weight 0.2 --constraint "RobustnessG(eps=0.1, delta=0.52)"
python mainDL2.py --dataset fashion_mnist --dtype baseline --dl2-weight 0.2 --constraint "LipschitzG(eps=0.1, L=10)"
python mainDL2.py --dataset fashion_mnist --dtype baseline --dl2-weight 0.2 --constraint "FGSM(eps=0.1, delta=10)"

python mainDL2.py --dataset gtsrb --dtype baseline --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainDL2.py --dataset gtsrb --dtype augmented --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainDL2.py --dataset gtsrb --dtype augmented_FGSM --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainFGSM.py --dataset gtsrb --dtype baseline --eps 0.1
python mainPGD.py --dataset gtsrb --dtype baseline --alfa 0 --beta 1
python mainDL2.py --dataset gtsrb --dtype baseline --dl2-weight 0.2 --constraint "TrueRobustness(eps=0.1, delta=10)"
python mainDL2.py --dataset gtsrb --dtype baseline --dl2-weight 0.2 --constraint "PseudoRobustness(eps=0.1)"
python mainDL2.py --dataset gtsrb --dtype baseline --dl2-weight 0.2 --constraint "RobustnessG(eps=0.1, delta=0.52)"
python mainDL2.py --dataset gtsrb --dtype baseline --dl2-weight 0.2 --constraint "LipschitzG(eps=0.1, L=10)"
python mainDL2.py --dataset gtsrb --dtype baseline --dl2-weight 0.2 --constraint "FGSM(eps=0.1, delta=10)"
```
To test Constraint Security and Constraint Likelihood of the models obtained by the experiments above, use the commands:
```
python testConstraintSecurity.py
python testConstraintLikelihood.py
```
