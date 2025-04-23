import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from torchinfo import summary
from torchviz import make_dot

import numpy as np
from timeit import default_timer as timer
import utils.params as params
from utils.elapse import elapse_time
from utils.resources import print_resources
from utils.datapipeline import import_data, get_dataset
from utils.model import get_model, get_lightning_module, BayesianLightningModule, GenericLightningModule
from utils.callback import callback_list
from utils.results import (calculate_uncertainty_trials, calculate_ece, perform_monte_carlo_sampling_trials,
                           plot_training_AccLoss, outputs2tensor, show_uncertainty_trials, print_cm)


# ==========================================================
## INTRO
# ==========================================================
start = timer()                 # start timer to calculate run time
GPUs, CPUs = print_resources()  # number of computation resources available
hparams = params.get_hparams()  # parse BASH run-time hyperparameters (used throughout script below)
params.save_hparams(hparams)    # create model folder and save hyperparameters list as .txt


# ==========================================================
## DEFINE LABELS
# ==========================================================
# class_names = ['Greedy', 'Greedy+', 'Auction', 'Auction+']; hparams.class_names=class_names # added as a .sh input; default is shown here
class_names = hparams.class_names
attribute_names = ["COMMS", "PRONAV"]; hparams.attribute_names=attribute_names
## IMPORT DATA (numpy objects)
x_train, y_train, x_test, y_test, cs_idx, input_shape, output_shape = import_data(hparams)
## CREATE DATALOADER OBJECTS
train_dataset, val_dataset, test_dataset = get_dataset(hparams, x_train, y_train, x_test, y_test)


# ==========================================================
## MODE: TRAIN or PREDICT
# ==========================================================
# Multi-GPU or CPU settings
if GPUs > 0:
    accelerator = "gpu"
    devices = "auto"    # Use all available GPUs
    strategy = "ddp" if GPUs > 1 else "auto"
else:
    accelerator = "cpu"
    devices = "auto"    # Use one CPU
    strategy = "auto"   # default
    # devices = CPUs    # Use all available CPUs
    # strategy = "ddp" if CPUs > 1 else None  # Use DDP for CPU parallelism if more than one CPU
# Initialize Logging and the Lightning Trainer
train_logger = CSVLogger(           # Creates training CSVLogger
    save_dir=hparams.model_dir,     # Directory to save logs
    name=None,                      # None saves metrics.csv directly into save_dir
    version="",                     # "" saves metrics.csv directly into save_dir
)
# test_logger=CSVLogger(              # Creates testing CSVLogger
#     save_dir=hparams.model_dir,     # Directory to save logs
#     name="test_predictions",        # saves metrics.csv into name dir
#     version="",                     # "" saves metrics.csv directly into name dir
# )
trainer = Trainer(
    max_epochs=hparams.num_epochs,
    accelerator=accelerator,
    devices=devices,
    strategy=strategy,
    callbacks=callback_list(hparams),   # early stopping and checkpoint/model saving (for training only)
    logger=train_logger,                # start with training CSVLogger
    log_every_n_steps=None,             # not logging training step metrics
    enable_progress_bar=False,          # Suppress outputs
    enable_model_summary=False,         # Suppress model summary
)
# TRAIN ***********************
if hparams.mode == 'train':
    print("\n*** TRAINING MODEL ***")
    train_timer = timer()               # start timer to calculate run time
    # Get the model (PyTorch Lightning Module)
    model = get_model(hparams, input_shape, output_shape)
    model = get_lightning_module(model, hparams)
    # Train the model
    trainer.fit(model, train_dataset, val_dataset)
    ## Time to train
    print(f"Training Time: {elapse_time(train_timer)}")
    ## Save training plot
    plot_training_AccLoss(hparams)
# PREDICT ***********************
else:
    print("\n*** LOAD PRETRAINED MODEL ***")
    if hparams.bnn: # Load Bayesian module
        model = BayesianLightningModule.load_from_checkpoint(hparams.trained_model)
    else:           # Load Generic module
        model = GenericLightningModule.load_from_checkpoint(hparams.trained_model)
    # Update the model's loaded params to reflect the current hparams, as required
    model.params.model_dir = hparams.model_dir                              # Allows saving to current directory
    model.params.num_monte_carlo = hparams.num_monte_carlo                  # Allows setting different then when model trained
    model.params.num_instances_visualize = hparams.num_instances_visualize  # Allows setting different then when model trained


# ==========================================================
## VISUALIZE MODEL
# ==========================================================
print("\n*** MODEL VISUALIZATION ***")
    # use TorchInfo package for textual description (better than Trainer model summary)
summary(model, input_size=(1, *input_shape))  # Needs batch size 1
    # use TorchViz package for graphic plot of model
dummy_input = torch.randn(1, *input_shape, requires_grad=True).to(model.device)
graph = make_dot(model(dummy_input), params=dict(model.named_parameters())) # generate graphic
graph.render(hparams.model_dir + "ModelGraph", format="pdf")                # save graphic


# ==========================================================
## TEST DATASET PREDICTIONS
# ==========================================================
print("\n*** TEST DATASET PREDICTIONS ***")
test_timer = timer()                    # start timer to calculate test inference
# Re-init trainer for test inference on ONE device (to avoid having to gather predictions across devices)
trainer = Trainer(
    accelerator=accelerator,
    devices=1,
    strategy="auto",
    callbacks=[],                       
    logger=train_logger,                # start with any CSVLogger and then update to None, or else throws TensorFlow error b/c Tensorboard default (even with None initially)
    log_every_n_steps=None,             # not logging training step metrics
    enable_progress_bar=False,          # Suppress outputs
    enable_model_summary=False,         # Suppress model summary
)
trainer.logger = None                   # switch off CSV output

# # TEST SET INFERENCE ***********************
# # Prints accuracy & loss, and stores outputs; conducts MC sampling for BNN
# trainer.test(model, test_dataset)
# # Move test results (predictions and labels) onto CPU for Numpy manipulation (N/A on GPU)
# predictions = model.test_predictions.cpu()
# labels = model.test_labels.cpu()
# print(f"Predictions shape: {predictions.shape}")
# print(f"Labels shape: {labels.shape}")
# print(f"Test Inference Time: {elapse_time(test_timer)}\n")

# TEST SET PREDICTION ***********************
prediction_outputs = trainer.predict(model, test_dataset)
predictions, labels, inputs = outputs2tensor(prediction_outputs)
if hparams.bnn:
    hparams.mc_sample_directory = "saved_mc_samples"
    hparams.mc_uncertainty_directory = "saved_uncertainty_metrics"
    num_trials = 10
    # ood_classes = [8,9]   # if only want to include left & down
    ood_classes = None
    perform_monte_carlo_sampling_trials(hparams, model, inputs, num_trials)
else:
    num_trials = None
    ood_classes = None
calculate_uncertainty_trials(hparams, predictions, labels, num_trials=num_trials)
calculate_ece(hparams, predictions, labels, num_bins=10, num_trials=num_trials)
show_uncertainty_trials(hparams, num_trials, ood_classes)
# Move test results (predictions and labels) onto CPU for Numpy manipulation (N/A on GPU)
predictions = predictions.cpu()
labels = labels.cpu()
print(f"\nPredictions shape: {predictions.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Test Inference Time: {elapse_time(test_timer)}\n")


# ==========================================================
## PRINT TEST DATASET SAMPLES: TRUE vs. PREDICTION
# ==========================================================
# convert prediction from probability (PDF) to integers; same for label(s)
if hparams.output_type == 'mc':             # Multiclass
    y_pred = predictions.argmax(dim=-1).numpy()     # Predicted class labels
    y_true = labels.numpy()                         # True class labels
elif hparams.output_type == 'ml':           # Multilabel
    y_pred = (predictions > 0.5).int().numpy()
    y_true = labels.numpy().astype(int)             # convert attribute true labels to integers
elif hparams.output_type == 'mh':           # Multihead
    pred_class, pred_attr = predictions             # Assume multihead outputs
    y_pred_class = pred_class.argmax(dim=-1).numpy()
    y_pred_attr = (pred_attr > 0.5).int().numpy()
    y_true_class, y_true_attr = labels              # Assume multihead labels
    y_true_class = y_true_class.numpy()
    y_true_attr = y_true_attr.numpy().astype(int)   # convert attribute true labels to integers

# print results to screen
num_results = 5  # Number of examples per class to display
print(f"\n*** TEST DATA RESULTS COMPARISON ({num_results} per class) ***")
unique_classes = torch.unique(labels, dim=0) if hparams.output_type != 'mh' else torch.unique(y_true_class)
if hparams.output_type != 'mh':
    for cls_num, true_label in enumerate(unique_classes):
        class_indices = ((labels == true_label).all(dim=1)).nonzero(as_tuple=True)[0]
        print(f"\nClass {cls_num} ({class_names[cls_num]}):")
        subset_indices = class_indices[:num_results]  # Select first `num_results` indices
        
        if hparams.output_type == 'mc':
            for idx in subset_indices:
                print(f"True: {int(y_true[idx].item())} | Predicted: {y_pred[idx]}")
        else:
            combo_str = ", ".join(
                f"{attribute_names[i]}={int(true_label[i].item())}" for i in range(len(attribute_names))
            )
            print(f"True Attributes: {combo_str}")
            for idx in subset_indices:
                true_values = " ".join(str(int(x.item())) for x in y_true[idx])
                pred_values = " ".join(str(int(x.item())) for x in y_pred[idx])
                print(f"True : {true_values} | Predicted : {pred_values}")
else:  # Multihead case
    for cls in unique_classes:
        class_indices = (y_true_class == cls).nonzero(as_tuple=True)[0]
        print(f"\nClass {cls.item()} ({class_names[cls.item()]}):")
        subset_indices = class_indices[:num_results]
        for idx in subset_indices:
            print(f"True Class: {y_true_class[idx]} | Predicted Class: {y_pred_class[idx]}")
            true_attrs_str = ", ".join(
                f"{attribute_names[i]}={y_true_attr[idx][i]}" for i in range(len(attribute_names))
            )
            pred_attrs_str = ", ".join(
                f"{attribute_names[i]}={y_pred_attr[idx][i]}" for i in range(len(attribute_names))
            )
            print(f"True Attributes: {true_attrs_str} | Predicted Attributes: {pred_attrs_str}")


# ==========================================================
## PRINT/SAVE CONFUSION MATRIX
# ==========================================================
print_cm(hparams, y_true, y_pred)



# ==========================================================
## PRINT TOTAL SCRIPT ELAPSE TIME
# ==========================================================
print(f"\nTotal Script Time: {elapse_time(start)}")
