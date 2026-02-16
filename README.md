# Intrusion-Detection-for-IoT-Networks
Bird-Inspired Optimization Approach using Taper-Shape Transfer Function for Intrusion Detection in IoT Networks

# 1. Introduction

The proliferation of Internet of Things (IoT) ecosystems has significantly increased the attack surface of modern cyber-physical systems. Due to the high dimensionality, heterogeneity, and dynamic characteristics of IoT traffic data, machine learning–based intrusion detection systems (IDS) often suffer from degraded classification performance and high computational complexity.

This repository provides the implementation of a bird-inspired metaheuristic feature selection (FS) framework integrated with a taper-shaped transfer function for binary optimization. The proposed approach aims to identify the most informative and non-redundant features from high-dimensional IoT datasets to enhance classification performance while reducing computational burden.

The framework is evaluated on:

RT-IoT2022

IoTID20

# 2. Methodological Framework
## 2.1 Feature Selection Formulation

The FS problem is modeled as a binary combinatorial optimization task:

Each feature is represented as a binary decision variable.

A taper-shaped transfer function is used to map continuous search agents into binary space.

The fitness function balances:

Classification performance

Feature reduction ratio

## 2.2 Optimization Algorithm

The primary optimization engine is:

Secretary Bird Optimization Algorithm (SBOA)

The algorithm is adapted to binary space using a transfer-function–based discretization strategy.

## 2.3 Classification Models

The selected feature subsets are evaluated using:

k-Nearest Neighbors (kNN)

Support Vector Machine (SVM)

Random Forest (RF)

All experiments are conducted using 10-fold cross-validation.

# 3. Experimental Setup
## 3.1 Datasets
Dataset	#Features	Type	Application Domain
RT-IoT2022	81	Real-time IoT traffic	Intrusion Detection
IoTID20	81	IoT Network Traffic	Intrusion Detection

Note: Due to dataset licensing restrictions, datasets are not redistributed in this repository. Please download them from their official sources.

## 3.2 Evaluation Metrics

The following performance metrics are computed:

Accuracy

Sensitivity (Recall)

Specificity

Precision

Feature Reduction Ratio (FRR)

Computational Time

# 4. Results Summary

## RT-IoT2022

Selected Features: 6 / 81

Feature Reduction Ratio: 92.59%

Accuracy: 99.69%

## IoTID20

Selected Features: 7 / 81

Feature Reduction Ratio: 91.36%

Accuracy: 98.46%

SBOA demonstrated strong robustness in handling high-dimensional IoT traffic data while achieving substantial dimensionality reduction.

# 5. Reproducibility

To ensure full reproducibility:

Random seeds are fixed.

10-fold cross-validation splits are deterministic.

Hyperparameters are reported in the manuscript.

All evaluation metrics are computed consistently across experiments.

# 7. Computational Environment

Experiments were conducted using:

Python 3.x

NumPy

Scikit-learn

Standard scientific computing libraries

Hardware details (recommended for replication):

Multi-core CPU

≥16GB RAM

# 8. Citation

If you use this repository in your research, please cite:
