import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import swarm.code.utils.params as params
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from timeit import default_timer as timer
import math

start = timer()
CPUs = os.cpu_count()
GPUs = len(tf.config.list_physical_devices('GPU'))
print(f"\nTensorflow Version: {tf.__version__}")
print(f"GPUs available: {GPUs}")
print(f"CPUs available: {CPUs}\n") 

hparams = params.get_hparams()
if not os.path.exists(hparams.model_dir):
    os.mkdir(hparams.model_dir)
params.save_hparams(hparams)


## IMPORT DATASET
# training and test data
data=np.load(hparams.data_path)
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
# real world test data
real_data=np.load(hparams.real_data_path)
x_real = real_data['x_test']
y_real = real_data['y_test']

# flatten timeseries data into 1 column per run for fully connected model
time_steps=x_train.shape[1]
num_features=x_train.shape[2]
num_inputs=time_steps*num_features
x_train=np.reshape(x_train,(len(x_train),num_inputs))
x_test=np.reshape(x_test,(len(x_test),num_inputs))
# real world has different number of time steps (more; will cut down in observation window so input fits into model)
real_time_steps=x_real.shape[1]
real_num_features=x_real.shape[2]
real_num_inputs=real_time_steps*real_num_features
x_real=np.reshape(x_real,(len(x_real),real_num_inputs))

print('\nxtrain shape:',x_train.shape)
print('xtest shape:',x_test.shape)
print('xreal shape:',x_real.shape)
print('ytrain shape:',y_train.shape)
print('ytest shape:',y_test.shape)
print('yreal shape:',y_real.shape)

print('\nx train sample (first instance, first 12 entries)',x_train[0,0:12])
print('y train sample (first instance)',y_train[0])

print('\nx real sample (first instance, first 12 entries)',x_real[0,0:12])
print('y real sample (first instance)',y_real[0])


## REDUCE OBSERVATION WINDOW
print('\nREDUCED OBSERVATION WINDOW')
min_time=time_steps
window=hparams.window
print('Min Run Time:',min_time)
print('Window:',window)
# if window = -1 or invalid window (too large>min_time): uses entire observation window (min_time for all runs)
if window!=-1 and window<min_time:
    num_inputs=window*num_features
else:
    num_inputs=min_time*num_features
x_train=x_train[:,:num_inputs]
x_test=x_test[:,:num_inputs]
x_real=x_real[:,:num_inputs]
print('Num Features:',num_features)
print('Num Inputs:',num_inputs)
print('xtrain shape:',x_train.shape)
print('xtest shape:',x_test.shape)
print('xreal shape:',x_real.shape,'\n')

## DEFINE MODEL
def build_model(
    input_shape,
    mlp_units,
    mlp_dropout=0
):
    inputs = keras.Input(shape=(input_shape,))
    x = inputs
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


## BUILD MODEL
n_classes = len(np.unique(y_train)) #number of classes = number of unique  labels
input_shape = num_inputs

model = build_model(
    input_shape,
    mlp_units=[100, 12], #100, 12
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()


## CALLBACKS
modelcheckpoint=tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(hparams.model_dir, "checkpoint{epoch:02d}-{val_loss:.2f}.h5"),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    save_freq=10*hparams.batch_size #once every 10 epochs; else 'epoch'
)

earlystopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=15, #15 or 50
    verbose=1,
    mode='min',
    restore_best_weights=True
)

## TRAIN MODEL
model_history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=hparams.num_epochs,
    batch_size=hparams.batch_size,
    verbose=0,
    callbacks=[earlystopping] #, modelcheckpoint] #may need to update TF version
)


## TRAINING CURVE
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
mvl=min(val_loss)
print(f"Minimum Val Loss: {mvl}") 

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig(hparams.model_dir + "/loss_vs_epoch.png")


## EVALUATE MODEL
pred=model.predict(x_test, verbose=0) #predicted label probabilities for test data
y_pred=np.argmax(pred,axis=1).reshape((-1,1))  #predicted label for test data
np.set_printoptions(precision=2) #show only 2 decimal places (does not change actual numbers)

# print('\nprediction & label:\n',np.hstack((pred,y_test))) #probability comparison
# print('\npredicted label & actual label:\n',np.hstack((y_pred,y_test))) #label comparison

eval=model.evaluate(x_test, y_test, verbose=0) #loss and accuracy
# print('\nevaluation:\n',eval) #print evaluation metrics numbers
print('\nmodel.metrics_names:\n',model.metrics_names) #print evaluation metrics names (loss and accuracy)
print(eval) #print evaluation metrics numbers

## CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.savefig(hparams.model_dir + "/conf_matrix.png")


## REAL WORLD ******************************
## EVALUATE MODEL
pred=model.predict(x_real, verbose=0) #predicted label probabilities for real world test data
y_pred=np.argmax(pred,axis=1).reshape((-1,1))  #predicted label for test data
np.set_printoptions(precision=2) #show only 2 decimal places (does not change actual numbers)

print('\nprediction & label:\n',np.hstack((pred,y_real))) #probability comparison
print('\npredicted label & actual label:\n',np.hstack((y_pred,y_real))) #label comparison

eval=model.evaluate(x_real, y_real, verbose=0) #loss and accuracy
print('\nmodel.metrics_names:\n',model.metrics_names) #print evaluation metrics names (loss and accuracy)
print(eval) #print evaluation metrics numbers

## CONFUSION MATRIX
cm = confusion_matrix(y_real, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.savefig(hparams.model_dir + "/conf_matrix_REAL.png")


## PRINT ELAPSE TIME
end = timer()
elapse=end-start
hours=0
minutes=0
seconds=0
remainder=0
if elapse>3600:
    hours=math.trunc(elapse/3600)
    remainder=elapse%3600
if elapse>60:
    if remainder>60:
        minutes=math.trunc(remainder/60)
        seconds=remainder%60
        seconds=math.trunc(seconds)
    else:
        minutes=math.trunc(elapse/60)
        seconds=elapse%60
        seconds=math.trunc(seconds)
if elapse<60:
    seconds=math.trunc(elapse)

print(f"\nElapse Time: {hours}h, {minutes}m, {seconds}s")