# Introduction

A Deep Learning-based EEG Decoding library for State-based Tasks/Analyses, using MNE and Tensorflow/Keras. 

This library is intended for EEG researchers and includes dataset processing tools, implementations of several cNN models, and scripts for model training, testing and visualization.

The aim of this project is to provide:

- a standard framework for EEG preprocessing and dataset extraction, suitable for machine learning analyses
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

# Deep Learning-based EEG Models

- cNN_3D [[1]](https://ieeexplore.ieee.org/abstract/document/9175324)
- cNN_topomap [[1]](https://ieeexplore.ieee.org/abstract/document/9175324)
- BrainDecode_Deep4 [[2]](https://arxiv.org/abs/1703.05051)
- BrainDecode_Shallow [[2]](https://arxiv.org/abs/1703.05051)
- EEGNet_v1 [[3]](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta)
- EEGNet_v2 [[3]](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta)

<img src="https://user-images.githubusercontent.com/17619349/193602946-85c0ee37-ecd5-4195-a940-3587ddc6cdc7.png" width=80% height=80%>
<img src="https://user-images.githubusercontent.com/17619349/193602960-9cd0a769-8796-4a76-bd4b-ba6cf163f24f.png" width=70% height=70%>

# Usage

To use this library, place the contents of the DL-EEG folder in your PYTHONPATH environment variable

# Citation

If you use this library for your research, please cite the following work:

```
@INPROCEEDINGS{9175324,
  author={Patlatzoglou, Konstantinos and Chennu, Srivas and Gosseries, Olivia and Bonhomme, Vincent and Wolff, Audrey and Laureys, Steven},
  booktitle={2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)}, 
  title={Generalized Prediction of Unconsciousness during Propofol Anesthesia using 3D Convolutional Neural Networks}, 
  year={2020},
  volume={},
  number={},
  pages={134-137},
  doi={10.1109/EMBC44109.2020.9175324}}
```
