import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.utils.util import predictive_entropy, mutual_information
import numpy as np
import pandas as pd
import os


## Deterministic NN Lightning Module
class GenericLightningModule(L.LightningModule):
    def __init__(self, model=None, params=None):
        super().__init__()
        self.model = model
        self.params = params

    def on_save_checkpoint(self, checkpoint):
        # Save custom attributes to the checkpoint to enable loading saved models
        checkpoint['model'] = self.model
        checkpoint['params'] = self.params

    def on_load_checkpoint(self, checkpoint):
        # Restore custom attributes from the checkpoint
        self.model = checkpoint.get('model', self.model) # Overwrite the init model with the trained DNN
        self.params = checkpoint.get('params', self.params)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)     # self(x) calls "forward" method; equivalent to self.model(x)
        loss = self._compute_loss(preds, y)
        class_acc, attr_acc = self._compute_accuracy(preds, y)
        if class_acc is not None:
            self.log("train_class_acc", class_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True) #sync_dist=True may slow down proccesses
        if attr_acc is not None:
            self.log("train_attr_acc", attr_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True) #sync_dist
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True) #sync_dist
        return loss     # is this needed?

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self._compute_loss(preds, y)
        class_acc, attr_acc = self._compute_accuracy(preds, y)
        if class_acc is not None:
            self.log("val_class_acc", class_acc, on_epoch=True, prog_bar=True, sync_dist=True) #sync_dist
        if attr_acc is not None:
            self.log("val_attr_acc", attr_acc, on_epoch=True, prog_bar=True, sync_dist=True) #sync_dist
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True) #sync_dist

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self._compute_loss(preds, y)
        class_acc, attr_acc = self._compute_accuracy(preds, y)
        if class_acc is not None:
            self.log("test_class_acc", class_acc, on_epoch=True, prog_bar=True)
        if attr_acc is not None:
            self.log("test_attr_acc", attr_acc, on_epoch=True, prog_bar=True)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return {"predictions": preds}  # is this needed?
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Check and initialize attributes if they do not exist
        if not hasattr(self, "test_predictions"):
            self.test_predictions = []          # Initialize list to collect predictions
        if not hasattr(self, "test_labels"):
            self.test_labels = []               # Initialize list to collect labels
        if not hasattr(self, "test_inputs"):
            self.test_inputs = []               # Initialize list to collect inputs
        # Collect predictions, labels, and inputs for later use
        x, y = batch
        self.test_predictions.append(outputs["predictions"].detach())  # detach: no longer track gradients for tensor
        self.test_labels.append(y.detach())     # Collect labels
        self.test_inputs.append(x.detach())     # Collect inputs

    def on_test_epoch_end(self):
        # Concatenate predictions, labels, and inputs after the epoch
        self.test_predictions = torch.cat(self.test_predictions, dim=0)     # convert list to torch tensor
        self.test_labels = torch.cat(self.test_labels, dim=0)               # convert list to torch tensor
        self.test_inputs = torch.cat(self.test_inputs, dim=0)               # convert list to torch tensor
                  
    def predict_step(self, batch, batch_idx):
        x, y = batch    # Use only x, as expected by forward()
        preds = self(x)
        return {"predictions": preds, "labels": y, "inputs": x}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.params.initial_learning_rate)

    def _compute_loss(self, preds, targets):
        if self.params.output_type == 'mh':  # Multihead output
            class_output, attr_output = preds
            class_target, attr_target = targets
            class_target = class_target.squeeze(-1)  # must have shape [batch] and not [batch,1]
            class_loss = F.nll_loss(torch.log(class_output + 1e-10), class_target)    # preds are softmax: take log with epsilon for numerical stability
            attr_loss = F.binary_cross_entropy(attr_output, attr_target)
            return class_loss + attr_loss #### NOTE: Multihead Loss Weights are equal (previous ML/MC=80/20)
        elif self.params.output_type == 'ml':  # Multilabel
            return F.binary_cross_entropy(preds, targets)
        elif self.params.output_type == 'mc':  # Multiclass
            targets = targets.squeeze(-1)                           # must have shape [batch] and not [batch,1]
            return F.nll_loss(torch.log(preds + 1e-10), targets)    # preds are softmax: take log with epsilon for numerical stability
        else:
            raise ValueError(f"Unsupported output_type: {self.params.output_type}")
        
    def _compute_accuracy(self, preds, targets):
        if self.params.output_type == 'mh':  # Multihead
            class_output, attr_output = preds
            class_target, attr_target = targets
            class_acc = (class_output.argmax(dim=-1) == class_target.squeeze(-1)).float().mean()
            attr_acc = ((attr_output > 0.5).float() == attr_target).float().mean()
            return class_acc, attr_acc
        else:  # Multiclass or Multilabel
            if self.params.output_type == 'ml':
                return None, ((preds > 0.5).float() == targets).float().mean()
            else:  # Multiclass
                # preds is typically a tensor of predicted probabilities or logits, with shape (batch_size, num_classes)
                # argmax(dim=-1) selects the index of the maximum value along the last dimension (num_classes); This gives the predicted class for each example in the batch
                # if targets has an extra singleton dimension (e.g., shape (batch_size, 1)), squeeze(-1) removes that dimension to ensure it matches the shape of preds.argmax(dim=-1)
                # (preds.argmax(dim=-1) == targets.squeeze(-1)) Returns a tensor of boolean values, where True indicates a correct prediction and False indicates an incorrect prediction
                # .float() Converts the boolean tensor into a float tensor, where True becomes 1.0 and False becomes 0.0
                # .mean() Computes the mean of the float tensor, effectively calculating the proportion of correct predictions (accuracy)
                return (preds.argmax(dim=-1) == targets.squeeze(-1)).float().mean(), None


## Bayesian NN Lightning Module
class BayesianLightningModule(GenericLightningModule):
    def __init__(self, model=None, params=None):
        super().__init__(model=model, params=params)
        
    def _compute_loss(self, preds, targets):
        # Total loss = NLL + KL
        loss = super()._compute_loss(preds, targets)                # Compute NLL loss
        kl_loss = get_kl_loss(self.model) / self.params.batch_size  # Add KL divergence loss for Bayesian model
        ######################### DEBUG PRINT
        # print(f"CE Loss: {loss}, KL Loss: {kl_loss}, Total Loss: {loss + kl_loss}")
        return loss + kl_loss


            
## DNN MODEL
def get_model(hparams, input_shape, output_shape):
    """
    Returns a deterministic torch.nn.Module based on the specified architecture.
    """
    model_type = hparams.model_type
    if model_type == 'fc':
        model = fc_model(hparams, input_shape, output_shape)
    elif model_type == 'cn':
        model = cnn_model(hparams, input_shape, output_shape)
    elif model_type == 'fcn':
        model = fcn_model(hparams, input_shape, output_shape)
    # elif model_type == 'res':
    #     model = resnet_model(hparams, input_shape, output_shape)
    elif model_type == 'lstm':
        model = lstm_model(hparams, input_shape, output_shape)
    elif model_type == 'tr':
        model = transformer_model(hparams, input_shape, output_shape)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model


## LIGHTNING MODULE
def get_lightning_module(model, hparams):
    """
    Returns appropriate PyTorch LIGHTNING Module (Bayesian or Generic).
    
    Args:
        model (torch.nn.Module): The deterministic model.
        hparams (Namespace): The parsed hyperparameters.
        
    Returns:
        LightningModule: Either a BayesianLightningModule or GenericLightningModule.
    """
    if hparams.bnn:
        # Bayesian prior parameters
        bnn_prior_parameters = {
            "prior_mu": 0.0,                # assumed gausian mean
            "prior_sigma": 0.1,             # assumed gausian std; WHY IS THIS 0.1 AND NOT 1.0 ?!?!?!
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Flipout",              # "Flipout" or "Reparameterization"; flipout is more efficient than reparm
            "moped_enable": os.path.exists(hparams.trained_model) if hparams.trained_model is not None else False,  # Enable MOPED if pretrained DNN model given
            "moped_delta": 0.5,
        }
        if bnn_prior_parameters["moped_enable"]:
            # Load the pretrained deterministic model
            checkpoint = torch.load(hparams.trained_model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            # Remove 'model.' prefix from keys (The state_dict saved by Lightning adds key prefix 'model.' because the LightningModule wraps your model)
            pretrained_state_dict = {key.replace("model.", ""): value for key, value in checkpoint["state_dict"].items()}
            model.load_state_dict(pretrained_state_dict)
        # Convert DNN to Bayesian model
        dnn_to_bnn(model, bnn_prior_parameters)

        return BayesianLightningModule(model, hparams)
    else:
        return GenericLightningModule(model, hparams)


## Multihead output model
class mh_version2(nn.Module):

    def __init__(self, prev_layer_output_size, output_shape):
        super(mh_version2, self).__init__()
        self.output_attr = nn.Linear(prev_layer_output_size, output_shape[1])  # Attribute predictions
        self.output_class = nn.Linear(prev_layer_output_size + output_shape[1], output_shape[0])  # Class predictions

    def forward(self, x):
        # Attribute output with sigmoid activation
        attr_output = torch.sigmoid(self.output_attr(x))
        # Concatenate x with attr_output
        concat = torch.cat((x, attr_output), dim=-1)
        # Class output with softmax activation
        class_output = F.softmax(self.output_class(concat), dim=-1)
        return [class_output, attr_output]



## FULLY CONNECTED (MULTILAYER PERCEPTRON)
class fc_model(nn.Module):

    def __init__(self, hparams, input_shape, output_shape):
        super(fc_model, self).__init__()
        self.output_type = hparams.output_type
        self.name = 'FC_' + self.output_type # Assign a name to the model
        # Determine MLP structure based on tuning
        if self.output_type == 'mh' and hparams.window == 20 and hparams.tuned:
            self.mlp_units = [60, 80]  # Tuned
            self.dropout_rate = 0.2  # Tuned
        else:
            self.mlp_units = [100, 12]  # Default structure
            self.dropout_rate = hparams.dropout
        self.input_layer = nn.Linear(input_shape, self.mlp_units[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(self.mlp_units) - 1):
            self.hidden_layers.append(nn.Linear(self.mlp_units[i], self.mlp_units[i + 1]))
        self.dropout = nn.Dropout(self.dropout_rate)
        # Determine output structure
        if self.output_type != 'mh':  # Single output
            self.output_layer = nn.Linear(self.mlp_units[-1], output_shape)
        else:  # Multihead outputs
            self.multihead_output = mh_version2(self.mlp_units[-1], output_shape)

    def forward(self, x):
        # Input layer processing
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        # Hidden layers processing
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        # Output layer
        if self.output_type != 'mh':  # Single output
            x = self.output_layer(x)
            # Apply appropriate activation
            if self.output_type == 'ml':  # Multilabel
                x = torch.sigmoid(x)
            elif self.output_type == 'mc':  # Multiclass
                x = F.softmax(x, dim=-1)
        else:  # Multihead outputs
            x = self.multihead_output(x)
        return x



## CONVOLUTIONAL
class cnn_model(nn.Module):

    def __init__(self, hparams, input_shape, output_shape):
        super(cnn_model, self).__init__()
        self.output_type = hparams.output_type
        self.name = 'CNN_' + self.output_type # Assign a name to the model
        # Determine CNN structure based on tuning
        if self.output_type == 'mc' and hparams.window == 20 and hparams.tuned:
            self.filters = [32]  # Tuned
            self.kernels = [7]  # Tuned
            self.pool_size = 5  # Tuned
            self.dropout_rate = 0.1  # Tuned
        elif self.output_type == 'mh' and hparams.window == 20 and hparams.tuned:
            self.filters = [224, 32, 160, 224, 96, 224]  # Tuned
            self.kernels = [7, 7, 5, 7, 5, 3]  # Tuned
            self.pool_size = 3  # Tuned
            self.dropout_rate = 0.3  # Tuned
        elif self.output_type == 'mh' and hparams.window == -1 and hparams.tuned:
            self.filters = [64, 32, 192, 96]  # Tuned
            self.kernels = [3, 3, 5, 7]  # Tuned
            self.pool_size = 3  # Tuned
            self.dropout_rate = 0.1  # Tuned
        else:  # Default structure
            self.filters = [64, 64, 64]
            self.kernels = [7, 5, 3]
            self.pool_size = 3
            self.dropout_rate = hparams.dropout
        self.stride = 2
        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        input_channels = input_shape[-1]  # input_shape = (sequence_length, num_channels)
        sequence_length = input_shape[0]  # Initial sequence length
        for filter, kernel in zip(self.filters, self.kernels):
            ######################### # DEBUG PRINT
            # print(f"FROM ZIP: Kernel size: {kernel}, Type: {type(kernel)}")
            self.conv_layers.append(nn.Conv1d(
                in_channels=input_channels,
                out_channels=filter,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,  # Equivalent to "same" padding
            ))
            input_channels = filter  # Update for next layer

        ######################### # DEBUG PRINT
        # for layer in self.conv_layers:
        #     print(f"FROM layer.kernel_size: Kernel size: {layer.kernel_size}, Type: {type(layer.kernel_size)}")

        # Define pooling layer
        self.pool = nn.MaxPool1d(kernel_size=self.pool_size, stride=self.stride, padding=0)
        # Update sequence length after pooling layers (each pooling layer reduces sequence length)
        for _ in self.conv_layers:
            sequence_length = (sequence_length - self.pool_size) // self.stride + 1  # Pooling reduces sequence length
            # print(f"Updated sequence length (after pooling): {sequence_length}")
        # Dynamically compute flattened size
        self.flattened_size = self.filters[-1] * sequence_length  # Total features after flatten layer (for matching up with output layer)
        # Define flatten and dropout
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(self.dropout_rate)
        # Define output layers
        if self.output_type != 'mh':  # Single output
            self.output_layer = nn.Linear(self.flattened_size, output_shape)
            #  self.output_layer = nn.Linear(self.filters[-1], output_shape)
        else:  # Multi-head outputs
            self.multihead_output = mh_version2(self.filters[-1], output_shape)
        
    #     # Apply custom HE Norm initialized weights
    #     self.apply(self._init_weights)

    # ## Attempting HE Normal weight init for better training performance (like TensorFlow)
    # def _init_weights(self, module):
    #     """
    #     Apply He normal initialization to Conv1d and Linear layers.
    #     """
    #     if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
    #         nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    #         if module.bias is not None:
    #             nn.init.constant_(module.bias, 0)

    def forward(self, x):
        ######################### # DEBUG PRINT
        # assert isinstance(x, torch.Tensor), f"cnn_model.forward: Expected tensor, got {type(x)}"
        # print(f"cnn_model.forward - Input shape: {x.shape}")

        # Permute to match PyTorch's Conv1D expectations (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        # Apply convolutional and pooling layers
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = self.pool(x)
        # Flatten and apply dropout
        # print(f"Shape after conv (before flatten): {x.shape}")
        x = self.flatten(x)
        # x = x.view(x.size(0), -1)  # Dynamically flatten
        # print(f"Shape after flatten (before dropout): {x1.shape}")
        # print(f"Shape after dynamic flatten (before dropout): {x.shape}")
        x = self.dropout(x)
        # print(f"Shape before output_layer: {x.shape}")
        # Output layer
        if self.output_type != 'mh':  # Single output
            x = self.output_layer(x)
            # Apply appropriate activation
            if self.output_type == 'ml':  # Multilabel
                x = torch.sigmoid(x)
            elif self.output_type == 'mc':  # Multiclass
                x = F.softmax(x, dim=-1)
        else:  # Multi-head outputs
            x = self.multihead_output(x)
        return x



## FULLY CONVOLUTIONAL
class fcn_model(nn.Module):

    def __init__(self, hparams, input_shape, output_shape):
        super(fcn_model, self).__init__()
        self.output_type = hparams.output_type
        self.name = 'FCN_' + self.output_type # Assign a name to the model
        # Determine FCN structure based on tuning
        if self.output_type == 'mh' and hparams.window == -1 and hparams.tuned:
            self.filters = [96, 32]  # Tuned
            self.kernels = [7, 5]  # Tuned
        else:
            self.filters = [64, 128, 256]  # Default structure
            self.kernels = [8, 5, 3]
        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        input_channels = input_shape[-1]  # Assuming input_shape = (sequence_length, num_channels)
        for filter, kernel in zip(self.filters, self.kernels):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=filter,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2  # Equivalent to "same" padding
                ),
                nn.BatchNorm1d(filter),
                nn.ReLU()
            ))
            input_channels = filter  # Update for next layer
        # Define final layers
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        if self.output_type != 'mh':  # Single output
            self.output_layer = nn.Conv1d(self.filters[-1], output_shape, kernel_size=1)
        else:  # Multi-head outputs
            self.multihead_output = mh_version2(self.filters[-1], output_shape)

    def forward(self, x):
        # Permute to match PyTorch's Conv1D expectations (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        # Apply global average pooling
        x = self.gap(x)  # Shape: (batch_size, channels, 1)
        x = x.squeeze(-1)  # Remove last dimension (batch_size, channels)
        # Output layer
        if self.output_type != 'mh':  # Single output
            x = self.output_layer(x.unsqueeze(-1))  # Add back sequence dimension for Conv1D
            x = x.squeeze(-1)  # Remove sequence dimension
            # Apply appropriate activation
            if self.output_type == 'ml':  # Multilabel
                x = torch.sigmoid(x)
            elif self.output_type == 'mc':  # Multiclass
                x = F.softmax(x, dim=-1)
        else:  # Multi-head outputs
            x = self.multihead_output(x)
        return x



## LONG SHORT TERM MEMORY (LSTM) WITH OPTIONAL MASKING
class lstm_model(nn.Module):
    def __init__(self, hparams, input_shape, output_shape):
        super(lstm_model, self).__init__()
        self.output_type = hparams.output_type
        self.name = 'LSTM_' + self.output_type # Assign a name to the model
        # Determine LSTM structure based on tuning
        if self.output_type == 'mc' and hparams.output_length == 'vec' and hparams.window == 20 and hparams.tuned:
            self.units = [100, 90, 60, 10, 10]
        elif self.output_type == 'mh' and hparams.output_length == 'seq' and hparams.window == -1 and hparams.tuned:
            self.units = [150]
        elif self.output_type == 'mh' and hparams.output_length == 'vec' and hparams.window == -1 and hparams.tuned:
            self.units = [120]
        else:
            self.units = [120, 90]  # Default structure
        self.dropout_rate = hparams.dropout
        self.output_length = hparams.output_length
        # Masking: PyTorch handles padded sequences with PackedSequence
        self.mask_value = 0.0  # Assume 0 is the padding value; adjust as necessary
        # Define LSTM layers
        self.lstm_layers = nn.ModuleList()
        input_size = input_shape[-1]  # Number of features in the input
        for i, unit in enumerate(self.units):
            self.lstm_layers.append(nn.LSTM(
                input_size=input_size,
                hidden_size=unit,
                batch_first=True,
                dropout=self.dropout_rate if i < len(self.units) - 1 else 0.0,  # No dropout on the last layer
                bidirectional=False
            ))
            input_size = unit  # Update input size for the next LSTM layer
        # Define output layers
        if self.output_type != 'mh':  # Single output
            self.output_layer = nn.Linear(self.units[-1], output_shape)
        else:  # Multi-head outputs
            self.multihead_output = mh_version2(self.units[-1], output_shape)

    def forward(self, x, lengths=None):
        # Masking: Convert inputs to PackedSequence if lengths are provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # Pass through LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
        # Unpack the sequence if packed
        if isinstance(x, nn.utils.rnn.PackedSequence):
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # Select output: sequence or vector
        if self.output_length == "vec":
            x = x[:, -1, :]  # Take the last time step for vector output
        # Output layer
        if self.output_type != 'mh':  # Single output
            x = self.output_layer(x)
            # Apply appropriate activation
            if self.output_type == 'ml':  # Multilabel
                x = torch.sigmoid(x)
            elif self.output_type == 'mc':  # Multiclass
                x = F.softmax(x, dim=-1)
        else:  # Multi-head outputs
            x = self.multihead_output(x)
        return x




## TRANSFORMER (ENCODER ONLY) WITH MASKING
class transformer_encoder(nn.Module):

    def __init__(self, dinput, num_heads, dff, dropout):
        super(transformer_encoder, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=dinput, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dinput)
        self.feed_forward = nn.Sequential(
            nn.Linear(dinput, dff),
            nn.ReLU(),
            nn.Linear(dff, dinput),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dinput)

    def forward(self, x, mask=None):
        # Self-Attention with optional Mask
        attn_output, _ = self.self_attention(x, x, x, key_padding_mask=mask) #input Q, K, V
        x = self.norm1(x + attn_output)
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x


class transformer_model(nn.Module):

    def __init__(self, hparams, input_shape, output_shape):
        super(transformer_model, self).__init__()
        self.output_type = hparams.output_type
        self.output_length = hparams.output_length
        self.name = 'TR_' + self.output_type  # Assign a name to the model
        # Determine Transformer structure based on tuning
        if hparams.output_type == 'mc' and self.output_length == 'vec' and hparams.window == 20 and hparams.tuned:
            self.num_enc_layers = 4
            self.dinput = 400
            self.dff = 400
            self.num_heads = 3
            self.dropout = 0.2
        elif hparams.output_type == 'mc' and self.output_length == 'vec' and hparams.window == -1 and hparams.tuned:
            self.num_enc_layers = 2
            self.dinput = 500
            self.dff = 400
            self.num_heads = 4
            self.dropout = 0.0
        elif hparams.output_type == 'mh' and self.output_length == 'seq' and hparams.window == -1 and hparams.tuned:
            self.num_enc_layers = 2
            self.dinput = 500
            self.dff = 600
            self.num_heads = 4
            self.dropout = 0.0
        else:
            self.num_enc_layers = 2
            self.dinput = 500
            self.dff = 400
            self.num_heads = 4
            self.dropout = hparams.dropout
        # Input embedding (Conv1d to match input feature dimension)
        self.input_embedding = nn.Conv1d(input_shape[-1], self.dinput, kernel_size=1, activation=nn.ReLU())
        # Positional Encoding
        self.positional_encoding = self.get_positional_encoding(length=2048, depth=self.dinput)
        # Transformer Encoder Layers
        self.encoders = nn.ModuleList([
            transformer_encoder(self.dinput, self.num_heads, self.dff, self.dropout)
            for _ in range(self.num_enc_layers)
        ])
        # Output Layer
        if self.output_type != 'mh':  # Single output
            self.output_layer = nn.Linear(self.dinput, output_shape)
        else:  # Multi-head outputs
            self.multihead_output = mh_version2(self.dinput, output_shape)

    def forward(self, x, mask=None):
        # Apply input embedding
        x = x.permute(0, 2, 1)  # Convert to (batch_size, features, sequence_length)
        x = self.input_embedding(x).permute(0, 2, 1)  # Back to (batch_size, sequence_length, features)
        # Apply positional encoding
        x += self.positional_encoding[:x.size(1), :]
        # Apply mask (convert to PyTorch format if provided)
        if mask is not None:
            mask = ~mask.bool()  # Invert mask for PyTorch (True = padded)
        # Pass through encoder layers
        for encoder in self.encoders:
            x = encoder(x, mask)
        # Reduce sequence dimension if required
        if self.output_type == 'vec':
            x = x.mean(dim=1)  # Global Average Pooling
        # Output Layer
        if self.output_type != 'mh':  # Single output
            x = self.output_layer(x)
            if self.output_type == 'ml':  # Multilabel
                x = torch.sigmoid(x)
            elif self.output_type == 'mc':  # Multiclass
                x = F.softmax(x, dim=-1)
        else:  # Multi-head outputs
            x = self.multihead_output(x)

        return x

    @staticmethod
    def get_positional_encoding(length, depth):
        if depth % 2 == 1: depth += 1  # Depth must be even
        positions = np.arange(length)[:, np.newaxis]  # (length, 1)
        depths = np.arange(depth // 2)[np.newaxis, :] / (depth // 2)  # (1, depth/2)
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates  # (length, depth/2)
        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)  # (length, depth)
        return torch.tensor(pos_encoding, dtype=torch.float32)
