from tensorflow import keras

def get_loss(hparams):
    if hparams.output_type == 'mc':
        loss="sparse_categorical_crossentropy"
    elif hparams.output_type == 'ml':
        loss="binary_crossentropy"
    elif hparams.output_type == 'mh':
        loss={'output_class':'sparse_categorical_crossentropy',
              'output_attr':'binary_crossentropy'}
    return loss


def get_optimizer(hparams):
    if hparams.optimizer == 'adam':
        optimizer=keras.optimizers.Adam(hparams.initial_learning_rate)
    elif hparams.optimizer == 'nadam':
        optimizer=keras.optimizers.Nadam(hparams.initial_learning_rate)
    return optimizer


def get_metric(hparams):
    if hparams.output_type == 'mc':
        metric="sparse_categorical_accuracy"
    elif hparams.output_type == 'ml':
        metric="binary_accuracy"
    elif hparams.output_type == 'mh':
        metric={'output_class':"sparse_categorical_accuracy",
                'output_attr':"binary_accuracy"}
    return metric