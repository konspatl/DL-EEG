# Introduction

A Deep Learning-based EEG Decoding library for State-based Tasks/Analyses, using MNE and Tensorflow/Keras. 

This library is intended for EEG researchers and includes dataset processing tools, implementations of several cNN models, and scripts for model training, testing and visualization.

The aim of this project is to provide:

- a standard framework for EEG preprocessing and dataset extraction, suitable for machine learning analysis
- the creation and training of deep learning models using EEG-driven network designs and methodologies
- the evaluation and testing of deep learning models over a variety of EEG systems and learning tasks
- a consistent EEG-processing pipeline, by exploiting the spatio-temporal structure of the EEG 
- improved research reproducibility and support for multi-study integration, using cross-subject validation schemes

# Requirements

- Python >= 3.6
- mne == 0.19
- tensorflow >= 2.6

Sample scripts requirements:

- numpy
- pathlib
- natsort
- pandas
- h5py
- scikit-learn

