import numpy as np
import tensorflow as tf
from timeit import default_timer as timer

from utils.elapse import elapse_time
from utils.resources import print_resources
import utils.params as params
from utils.datapipeline import import_data, get_dataset
from utils.model import get_model
from utils.compiler import get_loss, get_optimizer, get_metric
from utils.callback import callback_list
from utils.results import print_results
from utils.trainplot import train_plot

## INTRO
start = timer() # start timer to calculate run time
GPUs=print_resources() # computation resources available
hparams = params.get_hparams() # parse BASH run-time hyperparameters (used throughout script below)
params.save_hparams(hparams) #create model folder and save hyperparameters list .txt


## IMPORT DATA
x_train, y_train, x_test, y_test, num_classes, cs_idx, input_shape, output_shape = import_data(hparams)
## CREATE DATASET OBJECTS (to allow multi-GPU training)
train_dataset, val_dataset, test_dataset = get_dataset(hparams, x_train, y_train, x_test, y_test)


## BUILD & COMPILE MODEL
if hparams.mode == 'train':
    loss_weights=None #  single output head
    if hparams.output_type == 'mh': # multihead output
        loss_weights={'output_class':0.2,'output_attr':0.8}
        print(f"Loss Weights: {loss_weights}\n")
    if GPUs>1: # Multi-GPU
        print(f"GPUs availale: {GPUs}, MULTI GPU TRAINING")
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
        with mirrored_strategy.scope():
            model = get_model(hparams, input_shape, output_shape)
            model.compile(
                loss=get_loss(hparams),
                optimizer=get_optimizer(hparams),
                metrics=get_metric(hparams),
                loss_weights=loss_weights)
    else: # single gpu
        model = get_model(hparams, input_shape, output_shape)
        model.compile(
            loss=get_loss(hparams),
            optimizer=get_optimizer(hparams),
            metrics=get_metric(hparams),
            loss_weights=loss_weights)

elif hparams.mode == 'predict':
    model = tf.keras.models.load_model(hparams.trained_model)

if hparams.model_type!='tr':
    ## VISUALIZE MODEL
    model.summary()
    # make GRAPHVIZ plot of model
    tf.keras.utils.plot_model(model, hparams.model_dir + "graphviz.png", show_shapes=True)


## TRANSFORMER TROUBLESHOOTING
# instance=train_dataset.take(1)
# # for element in instance:
# #     print(f'train instance {element}')
# for (x, y), z in instance:
#   break
# print(f'x shape {x.shape}')
# print(f'y shape {y.shape}')
# print(f'z shape {z.shape}')
# output=model((x,y))
# print(f'output shape {output.shape}\n')
# ## VISUALIZE MODEL
# model.summary()
# # # make GRAPHVIZ plot of model
# # tf.keras.utils.plot_model(model, hparams.model_dir + "graphviz.png", show_shapes=True)
# # example=train_dataset.take(3)
# # for element in example:
# #     print(f'train element {element}')


## TRAIN MODEL
if hparams.mode == 'train':

    # # USING DATA
    # model_history = model.fit(
    #     x_train,
    #     y_train,
    #     validation_split=hparams.train_val_split,
    #     epochs=hparams.num_epochs,
    #     batch_size=hparams.batch_size,
    #     verbose=0,
    #     callbacks=callback_list(hparams)
    #     )

    # USING DATASET
    

    model_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=hparams.num_epochs,
        verbose=0,
        callbacks=callback_list(hparams)
        )
    ## PRINT HISTORY KEYS
    print("\nhistory.history.keys()\n", model_history.history.keys()) # this is what you want (only key names)
    ## PRINT TRAINING ELAPSE TIME
    elapse_time(start)
    if hparams.model_type!='tr':
        ## SAVE ENTIRE MODEL AFTER TRAINING
        model.save(filepath=hparams.model_dir+"model.keras", save_format="keras") #saves entire model: weights and layout
    ## TRAINING CURVE: ACCURACY/LOSS vs. EPOCH
    train_plot(hparams, model_history)

if hparams.model_type=='tr':
    ## VISUALIZE MODEL
    model.summary()
    # make GRAPHVIZ plot of model
    tf.keras.utils.plot_model(model, hparams.model_dir + "graphviz.png", show_shapes=True)


## TEST DATA PREDICTIONS
if hparams.output_type != 'mh':
    # pred=model.predict(x_test, verbose=0) #predicted label probabilities for test data
    pred=model.predict(test_dataset, verbose=0) #predicted label probabilities for test data
else: # multihead
    pred_class, pred_attr=model.predict(x_test, verbose=0) # multihead outputs 2 predictions (class and attribute)
# check for errors in requested output_length
output_length=hparams.output_length
if hparams.model_type!='lstm' and hparams.model_type!='tr': # only 'lstm' and 'tr' can output sequences
    output_length='vec'
# convert from probability to label
if hparams.output_type == 'mc' and output_length == 'vec':
    y_pred=np.argmax(pred,axis=1).reshape((-1,1))  #predicted class label for test data
elif hparams.output_type == 'mc' and output_length == 'seq':
    y_pred=np.argmax(pred,axis=2)  #predicted class label for test data
elif hparams.output_type == 'ml':
    y_pred=pred.round().astype(np.int32)  #predicted attribute label for test data
elif hparams.output_type == 'mh':
    y_pred_attr=pred_attr.round().astype(np.int32)
    if output_length == 'vec':
        y_pred_class=np.argmax(pred_class,axis=1).reshape((-1,1))
    else:
        y_pred_class=np.argmax(pred_class,axis=2)
    y_pred=[y_pred_class,y_pred_attr]
        

## TEST DATA PREDICTION SAMPLES
# np.set_printoptions(precision=2) #show only 2 decimal places for probability comparision (does not change actual numbers)
# print('\nprediction & label:\n',np.hstack((pred,y_test))) #probability comparison
num_results=2 # nunber of examples to view
print(f'\n*** TEST DATA RESULTS COMPARISON ({num_results} per class) ***')
print('    LABELS\nTrue vs. Predicted')
class_names = ['Greedy', 'Greedy+', 'Auction', 'Auction+']
if hparams.output_type != 'mh':
    for c in range(len(cs_idx)): # print each class name and corresponding prediction samples
        print(f"\n{class_names[c]}")
        # print(f"first true label: \n{y_test[cs_idx[c]]}")
        print(np.concatenate((y_test[cs_idx[c]:cs_idx[c]+num_results],
                            y_pred[cs_idx[c]:cs_idx[c]+num_results]), axis=-1))
else: # multihead output
    for c in range(len(cs_idx)): # print each class name and corresponding prediction samples
        print(f"\n{class_names[c]}")
        print('class')
        print(np.concatenate((y_test[0][cs_idx[c]:cs_idx[c]+num_results],
                            y_pred_class[cs_idx[c]:cs_idx[c]+num_results]), axis=-1))
        print('attribute')
        print(np.concatenate((y_test[1][cs_idx[c]:cs_idx[c]+num_results],
                            y_pred_attr[cs_idx[c]:cs_idx[c]+num_results]), axis=-1))


## EVALUATE MODEL
# eval=model.evaluate(x_test, y_test, verbose=0) #loss and accuracy
eval=model.evaluate(test_dataset, verbose=0) #loss and accuracy
print('model.metrics_names:\n',model.metrics_names) #print evaluation metrics names (loss and accuracy)
print(eval) #print evaluation metrics numbers


## RESULTS
print_results(hparams, y_test, y_pred)


## PRINT ELAPSE TIME
elapse_time(start)