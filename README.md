# Introduction

A Deep Learning-based EEG (DL-EEG) decoding library for state-based tasks/analyses, using MNE and Tensorflow/Keras. 

This library is intended for EEG researchers and includes dataset processing tools, implementations of several cNN models, and scripts for model training, testing and visualization.

**The aim of this project is to provide:**

- a standard framework for EEG preprocessing and dataset extraction, suitable for machine learning analyses
- methods for creation and training of deep learning models that employ EEG-driven network designs and methodologies
- methods for evaluation and testing of deep learning models over a variety of EEG systems and learning tasks
- a consistent EEG-processing pipeline, by exploiting the spatio-temporal structure of the EEG 
- support for multi-study integration and cross-subject/cross-study validation schemes, aiming at improved research reproducibility

# Requirements

* Python >= 3.6
* mne == 0.19
* tensorflow >= 2.6
- numpy
- scipy
- natsort
- matplotlib
- pandas
- scikit-learn


# Deep Learning-based EEG Models

The following deep learning models are implemented in the library (utils.NN):

- cNN_3D [[1]](https://ieeexplore.ieee.org/abstract/document/9175324)
- cNN_topomap [[1]](https://ieeexplore.ieee.org/abstract/document/9175324)
- BrainDecode_Deep4 [[2]](https://arxiv.org/abs/1703.05051)
- BrainDecode_Shallow [[2]](https://arxiv.org/abs/1703.05051)
- EEGNet_v1 [[3]](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta)
- EEGNet_v2 [[3]](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta)

<img src="https://user-images.githubusercontent.com/17619349/194333750-561ac1eb-3467-4dca-af66-e2db185646bc.png" width=80% height=80%>
<img src="https://user-images.githubusercontent.com/17619349/193602960-9cd0a769-8796-4a76-bd4b-ba6cf163f24f.png" width=60% height=60%>

# Usage

To use this library, place the contents of the DL-EEG folder in your PYTHONPATH environment variable

# Citation

If you use this library for your research, please cite the following work:

```
@phdthesis{kar97272,
           month = {September},
           title = {Deep Learning for Electrophysiological Investigation and Estimation of Anesthetic-Induced Unconsciousness},
          school = {University of Kent,},
          author = {Konstantinos Patlatzoglou},
            year = {2022},
        keywords = {deep learning EEG anesthesia consciousness},
             url = {https://kar.kent.ac.uk/97272/}
}
```
