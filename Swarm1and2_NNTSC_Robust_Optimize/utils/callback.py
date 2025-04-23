import tensorflow as tf

def callback_list(hparams):
    callbacks = []
    if "checkpoint" in hparams.callback_list:
        callbacks.append(_model_checkpoint(hparams))
    if "csv_log" in hparams.callback_list:
        callbacks.append(_csvlog(hparams))
    if "early_stopping" in hparams.callback_list:
        callbacks.append(_earlystopping(hparams))
    return callbacks


def _model_checkpoint(hparams):
    checkpoint=tf.keras.callbacks.ModelCheckpoint(
        filepath=hparams.model_dir + "checkpoint.h5",
        # filepath=hparams.model_dir + "checkpoint.keras", # doesn't work b/c ['options'] bug
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False, #full model is saved (model.save(filepath))
        mode='auto'
        )
    return checkpoint


def _csvlog(hparams):
    csv_log = tf.keras.callbacks.CSVLogger(filename=hparams.model_dir + "log.csv", append=True, separator=";")
    return csv_log


def _earlystopping(hparams):
    earlystopping=tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=hparams.patience, #15 or 50
        verbose=1,
        mode='min',
        restore_best_weights=True
        )
    return earlystopping
