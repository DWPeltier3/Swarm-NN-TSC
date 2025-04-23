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

def other_parameters(hparams):
    kernel_initializer=hparams.kernel_initializer
    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    if hparams.output_type == 'mc': # multiclass
        out_activation="softmax"
    elif hparams.output_type == 'ml': # multilabel
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]

    return kernel_initializer, kernel_regularizer, out_activation

## FULLY CONNECTED (MULTILAYER PERCEPTRON)
def fc_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    # MHw20 Tuned:
    if hparams.output_type=='mh' and hparams.window==20 and hparams.tuned:
        mlp_units=[60,80] # Tuned
        dropout=0.2 # Tuned
    else:
        mlp_units=[100,12] # each entry becomes a dense layer with corresponding # neurons (# entries = # hidden layers)
        dropout=hparams.dropout
    kernel_initializer, kernel_regularizer, out_activation = other_parameters(hparams)
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
        outputs=mh_version2(output_shape, out_activation, x) ## VERSION 2

    return keras.Model(inputs=inputs, outputs=outputs, name="Fully_Connected_" + hparams.output_type)


## CONVOLUTIONAL
def cnn_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    #MCw20 Tuned:
    if hparams.output_type=='mc' and hparams.window==20 and hparams.tuned:
        filters=[32] # Tuned
        kernels=[7] # Tuned
        pool_size=5 # Tuned
        dropout=0.1 # Tuned
    #MHw20 Tuned:
    elif hparams.output_type=='mh' and hparams.window==20 and hparams.tuned:
        filters=[224,32,160,224,96,224] # Tuned
        kernels=[7,7,5,7,5,3] # Tuned
        pool_size=3 # Tuned
        dropout=0.3 # Tuned
    # MHwFULL Tuned:
    elif hparams.output_type=='mh' and hparams.window==-1 and hparams.tuned:
        filters=[64,32,192,96] # Tuned
        kernels=[3,3,5,7] # Tuned
        pool_size=3 # Tuned
        dropout=0.1 # Tuned
    #CNNex (expanded capacity CNN for combined decoy motions and numbers)
    else:
        filters=[64,64,64] # each entry becomes a Conv1D layer (# entries = # conv layers)
        kernels=[7,5,3] # corresponding kernel size for Conv1d layer above
        pool_size=3 # max pooling window
        dropout=hparams.dropout
    stride=2
    padding="same" # "same" keeps output size = input size with padding
    kernel_initializer, kernel_regularizer, out_activation = other_parameters(hparams)
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for (filter, kernel) in zip(filters, kernels):
         x = keras.layers.Conv1D(activation="relu", filters=filter, kernel_size=kernel,
                                 padding=padding, kernel_regularizer=kernel_regularizer,
                                 kernel_initializer=kernel_initializer)(x)
         x = keras.layers.MaxPooling1D(pool_size=pool_size, strides=stride, padding=padding)(x)
    flat = keras.layers.Flatten()(x)
    drop = keras.layers.Dropout(dropout)(flat)
    if hparams.output_type!='mh':
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(drop)
    else: # multihead classifier (2 outputs)
        outputs=mh_version2(output_shape, out_activation, drop) ## VERSION 2

    return keras.Model(inputs=inputs, outputs=outputs, name='CNN_' + hparams.output_type)


## FULLY CONVOLUTIONAL
def fcn_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    # MHwFULL Tuned:
    if hparams.output_type=='mh' and hparams.window==-1 and hparams.tuned:
        filters=[96,32] # Tuned
        kernels=[7,5] # Tuned
    else:
        filters=[64,128,256] # each entry becomes a Conv1D layer (# entries = # conv layers)
        kernels=[8,5,3] # corresponding kernel size for Conv1d layer above
    padding="same" # "same" keeps output size = input size with padding
    kernel_initializer, kernel_regularizer, out_activation = other_parameters(hparams)
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
        outputs=mh_version2(output_shape, out_activation, gap) ## VERSION 2

    return keras.Model(inputs=inputs, outputs=outputs, name='FCN_' + hparams.output_type)


## LONG SHORT TERM MEMORY (LSTM) ** ADDED MASKING (single line of code)
def lstm_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    # MCVw20 Tuned:
    if hparams.output_type=='mc' and hparams.output_length=='vec' and hparams.window==20 and hparams.tuned:
        units=[100,90,60,10,10]
    # MHSwFULL Tuned:
    elif hparams.output_type=='mh' and hparams.output_length=='seq' and hparams.window==-1 and hparams.tuned:
        units=[150]
    # MHVwFULL Tuned:
    elif hparams.output_type=='mh' and hparams.output_length=='vec' and hparams.window==-1 and hparams.tuned:
        units=[120]
    else:
        units=[120,90] # each entry becomes an LSTM layer
    dropout=hparams.dropout
    kernel_initializer, kernel_regularizer, out_activation = other_parameters(hparams)
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    ## Adding a Masking layer (in case inputs are different dimensions)
    # Assume that 0 is the value used for padding; adjust if using a different padding value
    x = keras.layers.Masking(mask_value=0.0)(x)
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
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(x)
    else: # multihead classifier (2 outputs)
        outputs=mh_version2(output_shape, out_activation, x) ## VERSION 2

    return keras.Model(inputs=inputs, outputs=outputs, name='LSTM_' + hparams.output_type)


## TRANSFORMER (ENCODER ONLY) ** ADDED MASKING
def tr_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    # MCVw20 Tuned:
    if hparams.output_type=='mc' and hparams.output_length=='vec' and hparams.window==20 and hparams.tuned:
        num_enc_layers = 4 # Tuned
        dinput = 400  # Tuned
        dff = 400 # Tuned
        num_heads = 3 # Tuned
        dropout=0.2
    # MCVwFULL Tuned:
    elif hparams.output_type=='mc' and hparams.output_length=='vec' and hparams.window==-1 and hparams.tuned:
        num_enc_layers = 2 # Tuned
        dinput = 500  # Tuned
        dff = 400 # Tuned
        num_heads = 4 # Tuned
        dropout=0
    # MHSwFULL Tuned:
    elif hparams.output_type=='mh' and hparams.output_length=='seq' and hparams.window==-1 and hparams.tuned:
        num_enc_layers = 2 # Tuned
        dinput = 500  # Tuned
        dff = 600 # Tuned
        num_heads = 4 # Tuned
        dropout=0
    else:
        num_enc_layers = 2 #4 number of encoder layers
        dinput = 500  # 128; dimension of input embedding (and therefore also TRAN embedding dimension)
        dff = 400 # 512
        num_heads = 4 #4 number of attention heads (simultaneous attention mechanisims in parallel)
        dropout=hparams.dropout
    time_length=input_shape[0] #used to truncate positional encoding to time lenth of input
    kernel_initializer, kernel_regularizer, out_activation = other_parameters(hparams)
    print(f'# encode layers: {num_enc_layers}')
    print(f'# heads: {num_heads}')
    print(f'Dimension_Input: {dinput}')
    print(f'Dimension_Feedforward: {dff}')
    ## MODEL
    inputs = keras.Input(shape=input_shape)
    ## MASK layer -- Assuming padding value is 0
    mask = tf.math.not_equal(inputs, 0)  # Outputs a boolean tensor
    mask = tf.reduce_any(mask, axis=-1)  # Reduce across features, shape: [batch, time_length]
    mask = mask[:, tf.newaxis, tf.newaxis, :]  # Expand dimensions for broadcasting, shape: [batch, 1, 1, time_length]
## INPUT EMBEDDING
    x = inputs
    x = tf.keras.layers.Conv1D(filters=dinput, kernel_size=1, activation="relu")(x)
    # x = tf.keras.layers.LSTM(dinput,return_sequences=True)(x)
    # x = tf.keras.layers.Dense(dinput, activation="relu")(x)
    ## TIME POSITION ENCODING
    x *= tf.math.sqrt(tf.cast(dinput, tf.float32))
    x += positional_encoding(length=2048, depth=dinput)[tf.newaxis, :time_length, :dinput]
    for _ in range(num_enc_layers):
        # x = transformer_encoder(x, dinput, num_heads, dff, dropout)
        ## MASK
        x = transformer_encoder(x, dinput, num_heads, dff, dropout, mask)
    if hparams.output_length == 'vec': # if single output, need to reduce time axis to 1
        x = tf.keras.layers.GlobalAveragePooling1D()(x) # averages features across all time steps
    x = tf.keras.layers.Dropout(dropout)(x)
    if hparams.output_type!='mh':
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(x)
    else: # multihead classifier (2 outputs)
        outputs=mh_version2(output_shape, out_activation, x) ## VERSION 2
    
    return keras.Model(inputs=inputs, outputs=outputs, name='TRANS_' + hparams.output_type)

def transformer_encoder(input, dinput, num_heads, dff, dropout, mask):
    ## SELF ATTENTION
    # attn_output = tf.keras.layers.MultiHeadAttention(
    #     key_dim=dinput, num_heads=num_heads, dropout=dropout)(input, input)
    ## SELF ATTENTION with MASK
    attn_output = tf.keras.layers.MultiHeadAttention(
        key_dim=dinput, num_heads=num_heads, dropout=dropout)(input, input, attention_mask=mask)
    x = tf.keras.layers.Add()([input, attn_output])
    x = tf.keras.layers.LayerNormalization()(x) #default: std norm across last axis (-1=features)
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


## RESNET (with CONSTANT filter/kernel in each residual unit layer)
def resnet_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    num_res_layers=3 # number of layers in each residual unit (RU)
    filters=[64,64,64,128,128,128,256,256,256] # each entry becomes an RU (# entries = # RU)
    kernels=[7,5,3,7,5,3,7,5,3] # corresponding kernel size for each RU above
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
            print("KERNEL (1D): ",kernel)
            print("skip: ",skip.shape)
            print("x original: ",x.shape)
            x=x+skip
            print("x = x + skip: ",x.shape,'\n')
         x = keras.layers.ReLU()(x)
         count+=1
    output = x

    return output
