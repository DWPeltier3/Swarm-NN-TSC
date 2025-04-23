from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


def callback_list(hparams):
    """
    Generates a list of callbacks based on hparams.callback_list.
    """
    callbacks = []

    if "checkpoint" in hparams.callback_list:
        callbacks.append(_model_checkpoint(hparams))
    if "early_stopping" in hparams.callback_list:
        callbacks.append(_earlystopping(hparams))

    return callbacks


def _model_checkpoint(hparams):
    """
    Creates a ModelCheckpoint callback for saving the best model.
    """
    checkpoint = ModelCheckpoint(
        dirpath=hparams.model_dir,  # Directory to save checkpoints
        filename="model",      # Base filename
        # filename="best_model-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",         # Metric to monitor
        save_top_k=1,               # Save only the best model
        mode="min",                 # Minimize the metric
        save_weights_only=False,    # Save the entire model
    )
    return checkpoint


def _earlystopping(hparams):
    """
    Creates an EarlyStopping callback to stop training when no improvement is seen.
    """
    early_stopping = EarlyStopping(
        monitor="val_loss",           # Metric to monitor
        min_delta=0.0001,             # Minimum change to qualify as an improvement
        patience=hparams.patience,    # Number of epochs to wait
        mode="min",                   # Minimize the monitored metric
        verbose=False,                # Print early stopping messages (val loss values on epoch)
    )
    return early_stopping
