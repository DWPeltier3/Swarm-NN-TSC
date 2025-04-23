import tensorflow as tf
from tensorflow import keras
import numpy as np

def get_model(hparams, input_shape, output_shape):

    model_type=hparams.model_type

    if model_type == 'fc':
        model=fc_model(hparams, input_shape, output_shape)
    elif model_type == 'cn':
        model=cnn_model(hparams, input_shape, output_shape)
    elif model_type == 'fcn':
        model=fcn_model(hparams, input_shape, output_shape)
    elif model_type == 'res':
        model=resnet_model(hparams, input_shape, output_shape)
    elif model_type == 'lstm':
        model=lstm_model(hparams, input_shape, output_shape)
    elif model_type == 'tr':
        model=tr_model(hparams, input_shape, output_shape)
        
    return model

def mh_version1(output_shape, out_activation, x):
    output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(x)
    output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(x)
    outputs=[output_class, output_attr]
    return outputs
def mh_version2(output_shape, out_activation, x):
    output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(x)
    concat=keras.layers.Concatenate()([x,output_attr]) #use attribute output to try and improve class output
    output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(concat)
    outputs=[output_class, output_attr]
    return outputs

## FULLY CONNECTED (MULTILAYER PERCEPTRON)
def fc_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    mlp_units=hparams.mlp_units
    dropout=hparams.dropout
    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    kernel_initializer=hparams.kernel_initializer
    if hparams.output_type == 'mc': # multiclass
        out_activation="softmax"
    elif hparams.output_type == 'ml': # multilabel
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation="relu", kernel_regularizer=kernel_regularizer,
                               kernel_initializer=kernel_initializer)(x)
        x = keras.layers.Dropout(dropout)(x)
    if hparams.output_type!='mh':
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(x)
    else: # multihead classifier (2 outputs)
        # outputs=mh_version1(output_shape, out_activation, x) ## VERSION 1
        outputs=mh_version2(output_shape, out_activation, x) ## VERSION 2

    return keras.Model(inputs=inputs, outputs=outputs, name="Fully_Connected_" + hparams.output_type)


## CONVOLUTIONAL
def cnn_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    filters=hparams.filters # each entry becomes a Conv1D layer (# entries = # conv layers)
    kernels=hparams.kernels # corresponding kernel size for Conv1d layer above
    pool_size=hparams.pool_size # max pooling window
    stride=2
    padding="same" # "same" keeps output size = input size with padding
    dropout=hparams.dropout
    kernel_initializer=hparams.kernel_initializer
    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    if hparams.output_type == 'mc':
        out_activation="softmax"
    elif hparams.output_type == 'ml':
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for (filter, kernel) in zip(filters, kernels):
         x = keras.layers.Conv1D(activation="relu", filters=filter, kernel_size=kernel,
                                 padding=padding, kernel_regularizer=kernel_regularizer,
                                 kernel_initializer=kernel_initializer)(x)
         x = keras.layers.MaxPooling1D(pool_size=pool_size, strides=stride, padding=padding)(x)
    flat = keras.layers.Flatten()(x) #(conv3)
    drop = keras.layers.Dropout(dropout)(flat)
    if hparams.output_type!='mh':
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(drop)
    else: # multihead classifier (2 outputs)
        # outputs=mh_version1(output_shape, out_activation, drop) ## VERSION 1
        outputs=mh_version2(output_shape, out_activation, drop) ## VERSION 2

    return keras.Model(inputs=inputs, outputs=outputs, name='CNN_' + hparams.output_type)


## FULLY CONVOLUTIONAL
def fcn_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    filters=hparams.filters # each entry becomes a Conv1D layer (# entries = # conv layers)
    kernels=hparams.kernels # corresponding kernel size for Conv1d layer above
    padding="same" # "same" keeps output size = input size with padding
    kernel_initializer=hparams.kernel_initializer
    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    if hparams.output_type == 'mc':
        out_activation="softmax"
    elif hparams.output_type == 'ml':
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for (filter, kernel) in zip(filters, kernels):
         x = keras.layers.Conv1D(activation="relu", filters=filter, kernel_size=kernel,
                                 padding=padding, kernel_regularizer=kernel_regularizer,
                                 kernel_initializer=kernel_initializer)(x)
         x = keras.layers.BatchNormalization()(x)
         x = keras.layers.ReLU()(x)
    gap = keras.layers.GlobalAveragePooling1D()(x)
    if hparams.output_type!='mh':
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(gap)
    else: # multihead classifier (2 outputs)
        # outputs=mh_version1(output_shape, out_activation, gap) ## VERSION 1
        outputs=mh_version2(output_shape, out_activation, gap) ## VERSION 2

    return keras.Model(inputs=inputs, outputs=outputs, name='FCN_' + hparams.output_type)


## RESNET (with CONSTANT filter/kernel in each residual unit layer)
def resnet_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    num_res_layers=hparams.num_res_layers # number of layers in each residual unit (RU)
    filters=hparams.filters # each entry becomes an RU (# entries = # RU)
    kernels=hparams.kernels # corresponding kernel size for each RU above
    if hparams.output_type == 'mc':
        out_activation="softmax"
    elif hparams.output_type == 'ml':
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for (filter, kernel) in zip(filters, kernels):
         x=res_unit(hparams, num_res_layers, filter, kernel, x)
    gap = keras.layers.GlobalAveragePooling1D()(x)
    if hparams.output_type!='mh':
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(gap)
    else: # multihead classifier (2 outputs)
        # outputs=mh_version1(output_shape, out_activation, gap) ## VERSION 1
        outputs=mh_version2(output_shape, out_activation, gap) ## VERSION 2

    return keras.Model(inputs=inputs, outputs=outputs, name='ResNet_' + hparams.output_type)

def res_unit(
        hparams,
        num_res_layers,
        filter,
        kernel,
        input
        ):
    ## PARAMETERS
    padding="same"
    kernel_initializer=hparams.kernel_initializer
    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    ## RESIDUAL UNIT
    x = input
    skip = keras.layers.Conv1D(activation="relu", filters=filter, kernel_size=1,
                                 padding=padding, kernel_regularizer=kernel_regularizer,
                                 kernel_initializer=kernel_initializer)(input)
    count=1
    for _ in range(num_res_layers):
         x = keras.layers.Conv1D(activation="relu", filters=filter, kernel_size=kernel,
                                 padding=padding, kernel_regularizer=kernel_regularizer,
                                 kernel_initializer=kernel_initializer)(x)
         x = keras.layers.BatchNormalization()(x)
         if count==num_res_layers: #on last filter add original input (skip formated=Conv1D, kernel=1) before Relu
            # print("KERNEL (1D): ",kernel)
            # print("skip: ",skip.shape)
            # print("x original: ",x.shape)
            x=x+skip
            # print("x = x + skip: ",x.shape,'\n')
         x = keras.layers.ReLU()(x)
         count+=1
    output = x

    return output


## LONG SHORT TERM MEMORY (LSTM)
def lstm_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    units=hparams.units # each entry becomes an LSTM layer
    kernel_initializer=hparams.kernel_initializer
    dropout=hparams.dropout
    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    if hparams.output_type == 'mc':
        out_activation="softmax"
    elif hparams.output_type == 'ml':
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    count=1
    return_sequences=True # must return sequences when linking LSTM layers together
    for unit in units:
        if count == len(units): # on last LSTM unit, determine output: sequence or vector
            if hparams.output_length == "seq":
                return_sequences=True # sequence output
            else:
                return_sequences=False # vector output
        x = keras.layers.LSTM(unit,return_sequences=return_sequences, dropout=dropout,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer)(x)
        count+=1
    if hparams.output_type!='mh':
        # time distributed (dense) layer improves by 1%, but messes up "vector" output size
        # outputs = keras.layers.TimeDistributed(keras.layers.Dense(output_shape, activation=out_activation))(x)
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(x)
    else: # multihead classifier (2 outputs)
        # outputs=mh_version1(output_shape, out_activation, x) ## VERSION 1
        outputs=mh_version2(output_shape, out_activation, x) ## VERSION 2

    return keras.Model(inputs=inputs, outputs=outputs, name='LSTM_' + hparams.output_type)


## TRANSFORMER (ENCODER ONLY)
def tr_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    length=input_shape[0]
    num_enc_layers = hparams.num_enc_layers # number of encoder layers
    dinput = hparams.dinput #128 or = input_shape[1]
    dff = hparams.dff # 512
    num_heads = hparams.num_heads # 8
    dropout=hparams.dropout
    if hparams.output_type == 'mc':
        out_activation="softmax"
    elif hparams.output_type == 'ml':
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # input enmbedding
    x = tf.keras.layers.Conv1D(filters=dinput, kernel_size=1, activation="relu")(x)
    # position encoding
    x *= tf.math.sqrt(tf.cast(dinput, tf.float32))
    x += positional_encoding(length=2048, depth=dinput)[tf.newaxis, :length, :dinput]
    for _ in range(num_enc_layers):
        x = transformer_encoder(x, dinput, num_heads, dff, dropout)
    if hparams.output_length == 'vec': # if single output, need to reduce time axis to 1
        x = tf.keras.layers.GlobalAveragePooling1D()(x) # averages features across all time steps
    x = tf.keras.layers.Dropout(dropout)(x)
    if hparams.output_type!='mh':
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(x)
    else: # multihead classifier (2 outputs)
        # outputs=mh_version1(output_shape, out_activation, x) ## VERSION 1
        outputs=mh_version2(output_shape, out_activation, x) ## VERSION 2
    
    return keras.Model(inputs=inputs, outputs=outputs, name='TRANS_' + hparams.output_type)

def transformer_encoder(input, dinput, num_heads, dff, dropout):
    ## SELF ATTENTION
    attn_output = tf.keras.layers.MultiHeadAttention(
        key_dim=dinput, num_heads=num_heads, dropout=dropout)(input, input)
    x = tf.keras.layers.Add()([input, attn_output])
    x = tf.keras.layers.LayerNormalization()(x) #std norm across last axis (-1=features)
    skip = x
    ## FEED FORWARD
    x = tf.keras.layers.Dense(dff, activation='relu')(x)
    x = tf.keras.layers.Dense(dinput)(x)
    ff_output = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Add()([skip, ff_output])
    x = tf.keras.layers.LayerNormalization()(x)
    return x

def positional_encoding(length, depth):
    if depth % 2 == 1: depth += 1  # depth must be even
    depth = depth/2 # halve: 1/2 for SIN and COS
    positions = np.arange(length)[:, np.newaxis]     # (length, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth/2)
    angle_rates = 1 / (10000**depths)         # (1, depth/2)
    angle_rads = positions * angle_rates      # (length, depth/2)
    pos_encoding = np.concatenate(
       [np.sin(angle_rads), np.cos(angle_rads)],axis=-1) #(length, depth)
    return tf.cast(pos_encoding, dtype=tf.float32)