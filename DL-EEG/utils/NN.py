# Author: Konstantinos Patlatzoglou <konspatl@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, MaxPooling3D, \
    Flatten, AveragePooling1D, AveragePooling2D, AveragePooling3D, Reshape, Activation, BatchNormalization, \
    DepthwiseConv2D, SeparableConvolution2D, Permute
from tensorflow.keras.regularizers import l1_l2, l1
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

def create_model(model_name, input_shape, sfreq, output_shape, classification=True):
    """
    Creates and returns a model based on the given parameter specifications

    Args:
        model_name (str): the name of the model
        input_shape (tuple): the input shape of an input instance
        sfreq (int): the EEG sampling frequency
        output_shape (tuple): the output shape of output instance
        classification (boolean): if True, model design includes a softmax output layer, otherwise linear (Regression)
    Returns:
        model (tensorflow.keras.Model): a tensorflow/keras model
    """
    model = None

    if model_name == 'Toy':
        model = toyModel(input_shape[0], output_shape[0], classification=classification)

    elif model_name == 'cNN_3D':
        model = cNN_3D(input_shape[0], input_shape[1], input_shape[2], sfreq, output_shape[0],
                       classification=classification)

    elif model_name == 'cNN_topomap':
        model = cNN_topomap(input_shape[0], input_shape[1], input_shape[2], sfreq, output_shape[0],
                            classification=classification)

    elif model_name == 'BD_Deep4':
        model = BD_Deep4(input_shape[0], input_shape[1], sfreq, output_shape[0], classification=classification)

    elif model_name == 'BD_Shallow':
        model = BD_Shallow(input_shape[0], input_shape[1], sfreq, output_shape[0], classification=classification)

    elif model_name == 'EEGNet_v1':
        model = EEGNet_v1(input_shape[0], input_shape[1], sfreq, output_shape[0], classification=classification)

    elif model_name == 'EEGNet_v2':
        model = EEGNet_v2(input_shape[0], input_shape[1], sfreq, output_shape[0], classification=classification)

    return model


# ----------------------------------------------------MODELS-------------------------------------------------------------

def toyModel(samples, no_output_units, classification=True):
    """
    A Toy Model for Testing

    Args:
        samples (int): number of EEG samples per instance
        no_output_units (int): number of output units (targets)
        classification (boolean): if True, apply softmax activation in output layer, else apply linear units (Regression)
    Returns:
        Model (tensorflow.keras.Model): a tensorflow/keras model
    """
    input_shape = (samples,)

    # Sequential Model
    main_input = Input(shape=input_shape, name='main_input')

    # Layer 1
    x = Dense(100, input_shape=input_shape)(main_input)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Output Layer
    x = Flatten()(x)

    if classification:
        outputs = Dense(no_output_units, activation='softmax', name='main_output')(x)
    else: # Regression
        outputs = []
        for i in range(no_output_units):
            output = Dense(1, name=('output_' + str(i)))(x)
            outputs.append(output)

    return Model(main_input, outputs, name='Toy')



def cNN_3D(no_channels_x, no_channels_y, no_samples, sfreq, no_output_units, classification=True):
    """
     An implementation of the 3D cNN model proposed in (Patlatzoglou et al, 2020), (Patlatzoglou, 2022)

    This model employs a 2D grid representation of the EEG channel locations. Such architecture enables the exploitation
    of the spatio-temporal dynamics of EEG, as well as the integration of a variety of systems, under a common and
    consistent processing model.

    Args:
        no_channels_x (int): number of EEG channels in x dim of 2D Grid
        no_channels_y (int): number of EEG channels in y dim of 2D Grid
        no_samples (int): number of EEG samples per instance
        sfreq (int): sampling frequency of EEG (hz)
        no_output_units (int): number of output units (targets)
        classification (boolean): if True, apply softmax activation in output layer, else apply linear units (Regression)
    Returns:
        Model (tensorflow.keras.Model): a tensorflow/keras model
    """
    input_shape = (no_channels_x, no_channels_y, no_samples, 1)
    temp_kernel_1 = round(0.05 * sfreq) # 1st temporal kernel (5 ms)
    temp_kernel_2 = round(0.1 * sfreq) # 2nd temporal kernel (10 ms)

    # Sequential Model
    main_input = Input(shape=input_shape, name='main_input')

    # Layer 1
    x = Conv3D(32, kernel_size=(1, 1, temp_kernel_1), strides=(1, 1, 1), input_shape=input_shape)(main_input)
    x = Activation(activation='relu')(x)
    x = Conv3D(64, kernel_size=(2, 2, temp_kernel_2), strides=(1, 1, 1))(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2))(x)
    x = Dropout(0.25)(x)

    # Layer 2
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output Layer
    if classification:
        outputs = Dense(no_output_units, activation='softmax', name='main_output')(x)
    else:  # Regression
        outputs = []
        for i in range(no_output_units):
            output = Dense(1, name=('output_' + str(i)))(x)
            outputs.append(output)

    return Model(main_input, outputs, name='cNN_3D')


def cNN_topomap(topomap_x, topomap_y, no_samples, sfreq, no_output_units, classification=True):
    """
    An implementation of the cNN topomap model proposed in (Patlatzoglou et al, 2022)

    This model employs a 2D topomap representation of the EEG channel activity as scalp images. Such architecture
    enables the exploitation of the spatio-temporal dynamics of EEG, as well as the integration of a variety of systems,
    under a common and consistent processing model.

    Args:
        topomap_x (int): topomap image dimension x
        topomap_y (int): topomap image dimension y
        no_samples (int): number of EEG samples per instance
        sfreq (int): sampling frequency of EEG (hz)
        no_output_units (int): number of output units (targets)
        classification (boolean): if True, apply softmax activation in output layer, else apply linear units (Regression)
    Returns:
        Model (tensorflow.keras.Model): a tensorflow/keras model
    """
    input_shape = (topomap_x, topomap_y, no_samples, 1)
    temp_kernel_1 = round(0.05 * sfreq)  # 1st temporal kernel (5 ms)
    temp_kernel_2 = round(0.1 * sfreq)  # 2nd temporal kernel (10 ms)

    # Sequential Model...
    main_input = Input(shape=input_shape, name='main_input')

    # Layer 1
    x = AveragePooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), input_shape=input_shape)(main_input)
    x = Conv3D(32, kernel_size=(1, 1, temp_kernel_1), strides=(1, 1, 1))(x)
    x = Activation(activation='relu')(x)
    x = Conv3D(64, kernel_size=(2, 2, temp_kernel_2), strides=(1, 1, 1))(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2))(x)
    x = Dropout(0.25)(x)

    # Layer 2
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output Layer
    if classification:
        outputs = Dense(no_output_units, activation='softmax', name='main_output')(x)
    else:  # Regression
        outputs = []
        for i in range(no_output_units):
            output = Dense(1, name=('output_' + str(i)))(x)
            outputs.append(output)

    return Model(main_input, outputs, name='cNN_topomap')


def BD_Deep4(no_channels, no_samples, sfreq, no_output_units, classification=True):
    """
    An implementation of the Deep4Net model proposed in (Schirrmeister et al, 2017)
     https://arxiv.org/abs/1703.05051
     https://braindecode.org/

    Args:
        no_channels (int): number of EEG channels per instance
        no_samples (int): number of EEG samples per instance
        sfreq (int): sampling frequency of EEG (hz)
        no_output_units (int): number of output units (targets)
        classification (boolean): if True, apply softmax activation in output layer, else apply linear units (Regression)
    Returns:
        Model (tensorflow.keras.Model): a tensorflow/keras model
    """
    input_shape = (no_channels, no_samples, 1)
    temp_kernel = round(0.04 * sfreq)  # Temporal Kernel
    pool_size = (1, 2)

    # Sequential Model
    main_input = Input(shape=input_shape, name='main_input')

    # Layer 1
    x = Conv2D(25, kernel_size=(1, temp_kernel), strides=(1, 1), input_shape=input_shape)(main_input)
    x = Conv2D(25, kernel_size=(no_channels, 1), strides=(1, 1), use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x = Activation(activation='elu')(x)
    x = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='same')(x)

    # Layer 2
    x = Dropout(0.5)(x)
    x = Conv2D(50, kernel_size=(1, temp_kernel), strides=(1, 1), use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x = Activation(activation='elu')(x)
    x = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='same')(x)

    # Layer 3
    x = Dropout(0.5)(x)
    x = Conv2D(100, kernel_size=(1, temp_kernel), strides=(1, 1), use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x = Activation(activation='elu')(x)
    x = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='same')(x)

    # Layer 4
    x = Dropout(0.5)(x)
    x = Conv2D(200, kernel_size=(1, temp_kernel), strides=(1, 1), use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x = Activation(activation='elu')(x)
    x = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='same')(x)

    # Output Layer
    x = Flatten()(x)

    if classification:
        outputs = Dense(no_output_units, activation='softmax', name='main_output')(x)
    else:  # Regression
        outputs = []
        for i in range(no_output_units):
            output = Dense(1, name=('output_' + str(i)))(x)
            outputs.append(output)

    return Model(main_input, outputs, name='BD_Deep4')


def BD_Shallow(no_channels, no_samples, sfreq, no_output_units, classification=True):
    """
    An implementation of the ShallowFBCSPNet model proposed in (Schirrmeister et al, 2017)
     https://arxiv.org/abs/1703.05051
     https://braindecode.org/

    Args:
        no_channels (int): number of EEG channels per instance
        no_samples (int): number of EEG samples per instance
        sfreq (int): sampling frequency of EEG (hz)
        no_output_units (int): number of output units (targets)
        classification (boolean): if True, apply softmax activation in output layer, else apply linear units (Regression)
    Returns:
        Model (tensorflow.keras.Model): a tensorflow/keras model
    """
    input_shape = (no_channels, no_samples, 1)
    temp_kernel = round(0.1 * sfreq)  # Temporal Kernel
    pool_size = (1, round(0.3 * sfreq))
    pool_strides = (1, round(0.06 * sfreq))

    # Sequential Model
    main_input = Input(shape=input_shape, name='main_input')

    # Layer 1
    x = Conv2D(40, kernel_size=(1, temp_kernel), strides=(1, 1), input_shape=input_shape)(main_input)
    x = Conv2D(40, kernel_size=(no_channels, 1), strides=(1, 1), use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x = Activation(activation=square)(x)
    x = AveragePooling2D(pool_size=pool_size, strides=pool_strides)(x)
    x = Activation(activation=log)(x)
    x = Dropout(0.5)(x)

    # Output Layer
    x = Flatten()(x)

    if classification:
        outputs = Dense(no_output_units, activation='softmax', name='main_output')(x)
    else:  # Regression
        outputs = []
        for i in range(no_output_units):
            output = Dense(1, name=('output_' + str(i)))(x)
            outputs.append(output)

    return Model(main_input, outputs, name='BD_Shallow')


def EEGNet_v1(no_channels, no_samples, sfreq, no_output_units, classification=True):
    """
    An implementation of the EEGNet v1 (original) model proposed in (Lawhern et al, 2018)
     https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
     https://github.com/vlawhern/arl-eegmodels

    Args:
        no_channels (int): number of EEG channels per instance
        no_samples (int): number of EEG samples per instance
        sfreq (int): sampling frequency of EEG (hz)
        no_output_units (int): number of output units (targets)
        classification (boolean): if True, apply softmax activation in output layer, else apply linear units (Regression)
    Returns:
        Model (tensorflow.keras.Model): a tensorflow/keras model
    """
    input_shape = (no_channels, no_samples, 1)
    temp_kernel_1 = round(0.25 * sfreq)  # 1st temporal kernel
    temp_kernel_2 = round(0.03 * sfreq)  # 2nd temporal kernel

    # Sequential Model
    main_input = Input(shape=input_shape, name='main_input')

    # Layer 1
    x = Conv2D(16, kernel_size=(no_channels, 1), strides=(1, 1), kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
               input_shape=input_shape)(main_input)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.25)(x)
    x = Permute((3, 2, 1))(x)

    # Layer 2
    x = Conv2D(4, kernel_size=(2, temp_kernel_1), strides=(2, 4), padding='same',
               kernel_regularizer=l1_l2(l1=0.0, l2=0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.25)(x)

    # Layer 3
    x = Conv2D(4, kernel_size=(8, temp_kernel_2), strides=(2, 4), padding='same',
               kernel_regularizer=l1_l2(l1=0.0, l2=0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.25)(x)

    # Output Layer
    x = Flatten()(x)

    if classification:
        outputs = Dense(no_output_units, activation='softmax', name='main_output')(x)
    else:  # Regression
        outputs = []
        for i in range(no_output_units):
            output = Dense(1, name=('output_' + str(i)))(x)
            outputs.append(output)

    return Model(main_input, outputs, name='EEGNet_v1')


def EEGNet_v2(no_channels, no_samples, sfreq, no_output_units, classification=True):
    """
    An implementation of the EEGNet v2 (Newest) model proposed in (Lawhern et al, 2018)
     https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
     https://github.com/vlawhern/arl-eegmodels

    Args:
        no_channels (int): number of EEG channels per instance
        no_samples (int): number of EEG samples per instance
        sfreq (int): sampling frequency of EEG (hz)
        no_output_units (int): number of output units (targets)
        classification (boolean): if True, apply softmax activation in output layer, else apply linear units (Regression)
    Returns:
        Model (tensorflow.keras.Model): a tensorflow/keras model
    """
    input_shape = (no_channels, no_samples, 1)
    temp_kernel_1 = round(0.500 * sfreq)  # 1st temporal kernel
    temp_kernel_2 = round(0.125 * sfreq)  # 2nd temporal kernel

    # Sequential Model
    main_input = Input(shape=input_shape, name='main_input')

    # Layer 1
    x = Conv2D(8, kernel_size=(1, temp_kernel_1), strides=(1, 1), padding='same', use_bias=False,
               input_shape=input_shape)(main_input)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((no_channels, 1), strides=(1, 1), padding='valid', depth_multiplier=2, use_bias=False,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D(pool_size=(1, 4), strides=(1, 4), padding='same')(x)
    x = Dropout(0.25)(x)

    # Layer 2
    x = SeparableConvolution2D(16, kernel_size=(1, temp_kernel_2), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D(pool_size=(1, 8), strides=(1, 8))(x)
    x = Dropout(0.25)(x)

    # Output Layer
    x = Flatten()(x)

    if classification:
        outputs = Dense(no_output_units, activation='softmax', name='main_output')(x)
    else:  # Regression
        outputs = []
        for i in range(no_output_units):
            output = Dense(1, name=('output_' + str(i)))(x)
            outputs.append(output)

    return Model(main_input, outputs, name='EEGNet_v2')


# ------------------------------------------------GENERAL UTIL METHODS--------------------------------------------------

def get_optimizer(optimizer_name, learning_rate=1):
    """
    Returns an Optimizer instance based on the provided name and learning rate parameters

    Args:
        optimizer_name (str): the name of the optimizer (tensorflow/keras support)
        learning_rate (int): the learning rate
    Returns:
        optimizer (tf.keras.optimizers.Optimizer): the Optimizer instance
    """
    optimizer = None

    if optimizer_name == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSpropr(learning_rate=learning_rate)
    elif optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'Adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer_name == 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'Adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer_name == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer_name == 'Ftrl':
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=learning_rate)
    else:
        print('Optimizer is not supported')

    return optimizer


def get_model(model_path):
    """
    Loads and returns a tensorflow/keras model from the given path

    Args:
        model_path (str): path to model file
    Returns:
        model (tensorflow.keras.Model): a tensorflow/keras model
    """
    model = load_model(model_path, compile=True)
    return model


# Load all checkpoint models from model_path (except model.tf)
def get_checkpoint_models(model_checkpoint_path):
    """
    Loads and returns a list of tensorflow/keras models from the given path (for each training epoch)

    Args:
        model_checkpoint_path (Path): path to model files
    Returns:
        checkpoint_models (list): List of checkpoint tensorflow/keras models
    """
    model_files = sorted([file for file in model_checkpoint_path.iterdir() if file.suffix == '.tf'])

    checkpoint_models = []
    for model_file in model_files:
        model_name = model_file.stem
        model = load_model(str(model_file), compile=True)
        checkpoint_models.append(model)

    return checkpoint_models


def get_model_memory_usage(model, batch_size):
    """
    Returns the amount of memory needed in GPU for the given model and batch size

    Args:
        model (tensorflow.keras.Model): a tensorflow/keras model
        batch_size (int): the used batch size
    Returns:
        gbytes (int): the required memory (in GB)
    """
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(l, batch_size)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def get_ModelCheckpoint(filename, save_freq='epoch'):
    """
    Tensorflow/Keras callback to save the model at a given frequency

    Args:
        filename (str): path to save the model (can contain formatting options, e.g. for epoch)
        save_freq (str or int): 'epoch' or int for saving after 'n' number of batches
    Returns:
        checkpoint (tf.keras.callbacks.ModelCheckpoint)
    """
    checkpoint = ModelCheckpoint(filename, save_freq=save_freq, monitor='loss', verbose=1, save_best_only=False,
                                 save_weights_only=False, mode='auto')
    return checkpoint

# -----------------------------------------------------------OTHER-------------------------------------------------------

def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))





