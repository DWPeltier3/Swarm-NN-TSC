from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,multilabel_confusion_matrix,hamming_loss,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
from utils.savevariables import append_to_csv

def print_train_plot(hparams, model_history):
    ## TRAINING CURVE: Loss & Accuracy vs. EPOCH
    epoch = np.array(model_history.epoch)
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    if hparams.output_type == 'mc':
        acc=model_history.history["sparse_categorical_accuracy"]
        val_acc = model_history.history['val_sparse_categorical_accuracy']
    elif hparams.output_type == 'ml':
        acc=model_history.history["binary_accuracy"]
        val_acc = model_history.history['val_binary_accuracy']
    
    mvl=min(val_loss)
    print(f"Minimum Val Loss: {mvl}") 

    plt.figure()
    # must shift TRAINING values 1/2 epoch left b/c validaiton values computed AFTER each epoch, while training values are averaged during epoch
    if hparams.output_type != 'mh':
        plt.plot(epoch-0.5, loss, 'r', label='Training Loss')
        plt.plot(epoch, val_loss, 'b', label='Validation Loss')
        plt.plot(epoch-0.5, acc, 'tab:orange', label='Training Accuracy')
        plt.plot(epoch, val_acc, 'g', label='Validation Accuracy')
    else:
        # color names found at https://matplotlib.org/stable/gallery/color/named_colors.html
        plt.plot(epoch-0.5, model_history.history['loss'], 'r', label='Training Loss')
        plt.plot(epoch-0.5, model_history.history['output_class_loss'], "mistyrose", label='Class Trng Loss')
        plt.plot(epoch-0.5, model_history.history['output_attr_loss'], "lightcoral", label='Attribute Trng Loss')
        
        plt.plot(epoch, model_history.history['val_loss'], 'b', label='Validation loss')
        plt.plot(epoch, model_history.history['val_output_class_loss'], "lightsteelblue", label='Class Val Loss')
        plt.plot(epoch, model_history.history['val_output_attr_loss'], "cornflowerblue", label='Attribute Val Loss')

        plt.plot(epoch-0.5, model_history.history['output_class_sparse_categorical_accuracy'], "bisque", label='Class Trng Accuracy')
        plt.plot(epoch-0.5, model_history.history['output_attr_binary_accuracy'], "darkorange", label='Attribute Trng Accuracy')

        plt.plot(epoch, model_history.history['val_output_class_sparse_categorical_accuracy'], "lightgreen", label='Class Val Accuracy')
        plt.plot(epoch, model_history.history['val_output_attr_binary_accuracy'], "green", label='Attribute Val Accuracy')

    plt.title('Training & Validation Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.savefig(hparams.model_dir + "Training_LossAccuracy_vs_Epoch.png")




def print_cm(hparams, y_test, y_pred):
    # multiclass
    if hparams.output_type == 'mc':
        if hparams.output_length == 'seq':
            y_test=y_test.reshape((-1,1)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred=y_pred.reshape((-1,1)) #flatten both predictions and labels into one column vector each
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=hparams.class_names)
        disp.plot()
        plt.savefig(hparams.model_dir + "conf_matrix_MC.png")
        print("\n",classification_report(y_test, y_pred, target_names=hparams.class_names))
    # multilabel
    elif hparams.output_type == 'ml':
        if hparams.output_length == 'seq':
            y_test=y_test.reshape((-1,2)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred=y_pred.reshape((-1,2)) #flatten both predictions and labels into 2 column vectors each
        
        # Generate confusion matrix FIGURE & PRINT to log
        num_labels = len(y_test[0])
        cm = multilabel_confusion_matrix(y_test, y_pred)
        print('\n*** CONFUSION MATRIX ***\n[TN FP]\n[FN TP]')
        for label_index in range(num_labels):
            disp = ConfusionMatrixDisplay(cm[label_index], display_labels=['No', 'Yes'])
            disp.plot()
            plt.title(f'{hparams.attribute_names[label_index]} Confusion Matrix')
            plt.savefig(hparams.model_dir + f"conf_matrix_ML_{hparams.attribute_names[label_index]}.png")
            print(f'\nLabel{label_index} {hparams.attribute_names[label_index]}\n',cm[label_index]) 
        print('\nHamming Loss:',hamming_loss(y_test, y_pred),'\n')
        print("\n",classification_report(y_test, y_pred, target_names=hparams.attribute_names))
    # multihead
    elif hparams.output_type == 'mh':
        # multiclass results
        if hparams.output_length == 'seq':
            y_test[0]=y_test[0].reshape((-1,1)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred[0]=y_pred[0].reshape((-1,1)) #flatten predictions and labels into column vectors
        cm = confusion_matrix(y_test[0], y_pred[0])
        disp = ConfusionMatrixDisplay(cm, display_labels=hparams.class_names)
        disp.plot()
        plt.savefig(hparams.model_dir + "conf_matrix_MC.png")
        print("\n",classification_report(y_test[0], y_pred[0], target_names=hparams.class_names))
        # multilabel results
        if hparams.output_length == 'seq':
            y_test[1]=y_test[1].reshape((-1,2)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred[1]=y_pred[1].reshape((-1,2)) #flatten predictions and labels into 2 column vectors
        # Generate confusion matrix FIGURE & PRINT to log
        num_labels = len(y_test[1][0])
        cm = multilabel_confusion_matrix(y_test[1], y_pred[1])
        print('\n*** CONFUSION MATRIX ***\n[TN FP]\n[FN TP]')
        for label_index in range(num_labels):
            disp = ConfusionMatrixDisplay(cm[label_index], display_labels=['No', 'Yes'])
            disp.plot()
            plt.title(f'{hparams.attribute_names[label_index]} Confusion Matrix')
            plt.savefig(hparams.model_dir + f"conf_matrix_ML_{hparams.attribute_names[label_index]}.png")
            print(f'\nLabel{label_index} {hparams.attribute_names[label_index]}\n',cm[label_index])
        print('\nHamming Loss:',hamming_loss(y_test[1], y_pred[1]),'\n')
        print("\n",classification_report(y_test[1], y_pred[1], target_names=hparams.attribute_names))




def print_cam(hparams, model, x_train):
    ''' This function  prints the "Class Activation Map" along with model inputs to help 
    visualize input importance in making a prediction.  This is only applicable for models
    that have a Global Average Pooling layer (ex: Fully Convolutional Network)'''
    
    # select one training sample (engagement) to analyze
    sample = x_train[6] # 6 = Auction+
    # Get the class activation map for that sample
    last_cov_layer=-5 if hparams.output_type == 'mh' else -3 # multihead v2 has 2 extra layers at end
    heatmap = get_cam(hparams, model, sample, model.layers[last_cov_layer].name)
    # Save CAM variable for postprocessing
    append_to_csv(heatmap, hparams.model_dir + "variables.csv", "\nCAM Data")
    ## Visualize Class Activation Map
    # ALL FEATURES: Plot the heatmap values along with the time series data for all features (all agents) in that sample
    plt.figure(figsize=(10, 8))
    plt.plot(sample, c='black')
    plt.plot(heatmap, label='CAM [importance]', c='red', lw=5, linestyle='dashed')
    plt.xlabel('Time Step')
    plt.ylabel('Feature Value [normalized]')
    plt.legend()
    plt.title(f'Class Activation Map vs. All Input Features')
    plt.savefig(hparams.model_dir + "CAM_all.png")
    # ONE AGENT: Plot the heatmap values along with the time series features for one agent in that sample
    plt.figure(figsize=(10, 8))
    agent_idx=0
    for feature in range(hparams.num_features_per):
        plt.plot(sample[:, agent_idx+feature*hparams.num_agents], label=hparams.feature_names[feature])
    plt.plot(heatmap, label='CAM [importance]', c='red', lw=5, linestyle='dashed')
    plt.xlabel('Time Step')
    plt.ylabel('Feature Value [normalized]')
    plt.legend()
    plt.title(f"Class Activation Map vs. One Agent's Input Features")
    plt.savefig(hparams.model_dir + "CAM_one.png")

def get_cam(hparams, model, sample, last_conv_layer_name):
    # This function requires the trained FCN model, input sample, and the name of the last convolutional layer
    # Get the model of the intermediate layers
    cam_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output[0]]
        )
    # Get the last conv_layer outputs and full model predictions
    with tf.GradientTape() as tape:
        conv_outputs, predictions = cam_model(np.array([sample]))
        predicted_class = tf.argmax(predictions[0]) # predicted class index
        predicted_class_val = predictions[:, predicted_class] # predicted class probability
    # Get the gradients and pooled gradients
    grads = tape.gradient(predicted_class_val, conv_outputs) # gradients between predicted class probability WRT CONV outputs maps
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1)) # average ("pool") feature gradients (gives single importance value for each map)
    # Multiply pooled gradients (importance) with the conv layer output, then average across all feature maps, to get the 2D heatmap
    # Heatmap highlights areas that most influence models prediction
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap) # normalize values [0,1]
    # heatmap = heatmap * np.max(sample) # scale to sample maximum
    # print(f"Heatmap shape {heatmap.shape} \n heatmap values \n {heatmap} ")
    # append_to_csv(heatmap, hparams.model_dir + "variables.csv", "\nHeatmap")
    return heatmap[0]





def print_tsne(hparams, tsne_input, labels, title, perplexity):
    tsne = TSNE(n_components=2, perplexity=perplexity).fit_transform(tsne_input) # set perplexity = 50-100
    scaler = MinMaxScaler() # scale between 0 and 1
    tsne = scaler.fit_transform(tsne.reshape(-1, tsne.shape[-1])).reshape(tsne.shape) # fit amongst tsne_input, then back to original shape
    
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    # Dynamically generate colors up to 10
    color_list = ['red', 'blue', 'green', 'brown', 'yellow', 'purple', 'orange', 'pink', 'gray', 'cyan']
    num_classes = len(hparams.class_names)
    
    if num_classes > len(color_list):
        raise ValueError(f"Number of classes ({num_classes}) exceeds the number of available colors ({len(color_list)}).")

    colors = color_list[:num_classes]

    plt.figure()
    plt.title('tSNE Dimensionality Reduction: '+title)
    
    for idx, c in enumerate(colors):
        indices = [i for i, l in enumerate(labels) if idx == l]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        plt.scatter(current_tx, current_ty, c=c, label=hparams.class_names[idx], alpha=0.5, marker=".")
    
    plt.legend(loc='best')
    plt.savefig(hparams.model_dir + "tSNE_"+title+".png")




def compute_and_print_saliency(hparams, model, dataset):
    # Initialize running sums for the metrics and sensitivity summaries
    running_total_sensitivity = 0
    running_sensitivity = 0
    running_overall_sensitivity = 0
    running_sensitivity_summary = None
    num_batches = 0

    # Iterate through the dataset batch by batch
    for inputs, labels in dataset:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            predictions = model(inputs)

        # Check if predictions are multi-head outputs
        if isinstance(predictions, list):
            grads = [tape.jacobian(pred, inputs) for pred in predictions]
            mean_grads = [tf.reduce_mean(g, axis=[0, 2]) for g in grads]  # Reduce along batch and second axis
            sensitivity_summaries = [tf.norm(mean_grad, axis=0) for mean_grad in mean_grads]
        else:
            grads = tape.jacobian(predictions, inputs)
            mean_grads = tf.reduce_mean(grads, axis=[0, 2])  # Reduce along batch and second axis
            sensitivity_summaries = tf.norm(mean_grads, axis=0)

        # Calculate metrics for each batch
        if isinstance(sensitivity_summaries, list):
            for i, sensitivity_summary in enumerate(sensitivity_summaries):
                batch_total_sensitivity = tf.reduce_sum(tf.abs(sensitivity_summary))
                batch_sensitivity = tf.reduce_sum(tf.abs(mean_grads[i]))
                batch_overall_sensitivity = tf.norm(sensitivity_summary)

                # Update running totals for multi-head case
                running_total_sensitivity += batch_total_sensitivity
                running_sensitivity += batch_sensitivity
                running_overall_sensitivity += batch_overall_sensitivity

                # Accumulate sensitivity summaries for plotting
                if running_sensitivity_summary is None:
                    running_sensitivity_summary = [tf.identity(sensitivity_summary) for sensitivity_summary in sensitivity_summaries]
                else:
                    running_sensitivity_summary[i] += sensitivity_summary

        else:
            batch_total_sensitivity = tf.reduce_sum(tf.abs(sensitivity_summaries))
            batch_sensitivity = tf.reduce_sum(tf.abs(mean_grads))
            batch_overall_sensitivity = tf.norm(sensitivity_summaries)

            # Update running totals for single-head case
            running_total_sensitivity += batch_total_sensitivity
            running_sensitivity += batch_sensitivity
            running_overall_sensitivity += batch_overall_sensitivity

            # Accumulate sensitivity summaries for plotting
            if running_sensitivity_summary is None:
                running_sensitivity_summary = tf.identity(sensitivity_summaries)
            else:
                running_sensitivity_summary += sensitivity_summaries

        num_batches += 1
        del grads, tape  # Manually delete to free memory

    # Average metrics over all batches
    avg_total_sensitivity = running_total_sensitivity / num_batches
    avg_sensitivity = running_sensitivity / num_batches
    avg_overall_sensitivity = running_overall_sensitivity / num_batches

    # Print or log the aggregated metrics
    print(f"Average Sum(abs(norm(mean(saliency)))): {avg_total_sensitivity}")
    print(f"Average Sum(abs(mean(saliency))): {avg_sensitivity}")
    print(f"Average Norm(mean(saliency)): {avg_overall_sensitivity}")

    # Plot sensitivity summary
    if isinstance(running_sensitivity_summary, list):
        # Multi-head output: Plot each head separately
        avg_sensitivity_summaries = [s / num_batches for s in running_sensitivity_summary]
        for i, avg_sensitivity_summary in enumerate(avg_sensitivity_summaries):
            sensitivity_array = avg_sensitivity_summary.numpy()
            time_steps = sensitivity_array.shape[0]
            num_features = sensitivity_array.shape[1]
            num_features_per = len(hparams.feature_names)
            num_agents = num_features // num_features_per

            plt.figure(figsize=(round(num_features / 2), round(time_steps / 2)))
            ax = sns.heatmap(sensitivity_array[np.newaxis, :], cmap="viridis", cbar_kws={'label': 'Sensitivity'}, square=True)
            ax.set_title(f'Model Sensitivity Across Input Features for Head {i}')
            ax.set_xlabel('Input Feature Group')
            ax.set_ylabel('Time Step')

            # Set the y-ticks to go from 1 to number of time steps
            y_ticks = np.arange(0.5, time_steps, 1)  # Start from 0.5 to position in the middle of each square
            ax.set_yticks(y_ticks)  # Set y-ticks to the center of each row
            ax.set_yticklabels(np.arange(1, time_steps + 1))  # Labels from 1 to number of time steps

            # Set the x-ticks to be the middle of each group of features
            ticks = np.arange(round(num_agents / 2), num_features, num_agents)
            labels = hparams.feature_names
            ax.set_xticks(ticks) # Apply custom ticks and labels
            ax.set_xticklabels(labels, rotation=45) # Rotate labels for better readability
            for j in range(num_agents, num_features, num_agents):
                ax.axvline(x=j, color='white', linestyle='-', linewidth=3)
            ax.invert_yaxis()

            plt.savefig(f"{hparams.model_dir}saliency_{hparams.model_type}{hparams.window}_head_{i}.png")

    else:
        # Single-head case: Plot the average sensitivity summary
        avg_sensitivity_summary = running_sensitivity_summary / num_batches
        sensitivity_array = avg_sensitivity_summary.numpy()
        time_steps = sensitivity_array.shape[0]
        num_features = sensitivity_array.shape[1]
        num_features_per = len(hparams.feature_names)
        num_agents = num_features // num_features_per

        plt.figure(figsize=(round(num_features / 2), round(time_steps / 2)))
        ax = sns.heatmap(sensitivity_array, cmap="viridis", cbar_kws={'label': 'Sensitivity'}, square=True)
        ax.set_title('Model Sensitivity Across Input Features')
        ax.set_xlabel('Input Feature Group')
        ax.set_ylabel('Time Step')

        # Set the y-ticks to go from 1 to number of time steps
        y_ticks = np.arange(0.5, time_steps, 1)  # Start from 0.5 to position in the middle of each square
        ax.set_yticks(y_ticks)  # Set y-ticks to the center of each row
        ax.set_yticklabels(np.arange(1, time_steps + 1))  # Labels from 1 to number of time steps

        # Set the x-ticks to be the middle of each group of 10 features
        ticks = np.arange(round(num_agents / 2), num_features, num_agents) # start/stop/step
        labels = hparams.feature_names
        ax.set_xticks(ticks) # Apply custom ticks and labels
        ax.set_xticklabels(labels, rotation=45) # Rotate labels for better readability
        for i in range(num_agents, num_features, num_agents):
            ax.axvline(x=i, color='white', linestyle='-', linewidth=3)
        ax.invert_yaxis()

        plt.savefig(f"{hparams.model_dir}saliency_{hparams.model_type}{hparams.window}.png")




# def compute_saliency(model, dataset):
#     saliency_maps = []
#     for inputs, labels in dataset:
#         with tf.GradientTape(persistent=True) as tape:
#             tape.watch(inputs)
#             predictions = model(inputs)
#         # Check if predictions is a list (multi-head output)
#         if isinstance(predictions, list):
#             # Compute Jacobian for each output in multi-head predictions
#             grads = [tape.jacobian(pred, inputs) for pred in predictions]
#         else:
#             grads = tape.jacobian(predictions, inputs)
#         saliency_maps.append(grads)
#         del tape  # Manually delete the tape to free resources
#     return saliency_maps

# def print_saliency(hparams, model, test_dataset):
#     # Compute Saliency
#     saliency = compute_saliency(model, test_dataset)

#     # Check if we have multi-head output by checking if saliency is a list of lists
#     is_multi_head = isinstance(saliency[0], list)

#     if is_multi_head:
#         # Multi-head case
#         averaged_saliency_heads = []

#         for head_saliency in saliency:
#             # Average each batch's saliency map before stacking
#             # The initial shape of each saliency map in the list: (batch_size, output_size, batch_size, time_steps, features)
#             averaged_saliency = [tf.reduce_mean(batch_saliency, axis=[0, 2]) for batch_saliency in head_saliency]  # Output Shape: (T, F)
#             # Stack the averaged saliency maps
#             saliency_tensor = tf.stack(averaged_saliency)  # Output Shape: (B, T, F)
#             # Average over the batch dimension
#             mean_saliency = tf.reduce_mean(saliency_tensor, axis=0)  # Output shape: (T, F)
#             averaged_saliency_heads.append(mean_saliency)

#         # Now 'averaged_saliency_heads' is a list of mean saliency maps for each head.
#         for i, mean_saliency in enumerate(averaged_saliency_heads):
#             # Summarize sensitivity by computing the norm across the output dimension
#             sensitivity_summary = tf.norm(mean_saliency, axis=0)  # Shape: (F)

#             ## Sum of Absolute Values
#             total_sensitivity = tf.reduce_sum(tf.abs(sensitivity_summary))
#             print(f"Sum(abs(norm(mean(saliency) for head {i}): ", total_sensitivity)
#             sensitivity = tf.reduce_sum(tf.abs(mean_saliency))
#             print(f"Sum(abs(mean(saliency) for head {i}): ", sensitivity)

#             ## Norm
#             overall_sensitivity = tf.norm(sensitivity_summary)
#             print(f"Norm(mean(saliency)) for head {i}: ", overall_sensitivity)

#             # Convert sensitivity summary to numpy array for plotting
#             sensitivity_array = sensitivity_summary.numpy()

#             # Find quantities to help with plotting
#             time_steps = sensitivity_array.shape[0]
#             num_features = sensitivity_array.shape[1]
#             num_features_per = len(hparams.feature_names)
#             num_agents = num_features // num_features_per

#             # Set up the plot dimensions and labels
#             plt.figure(figsize=(round(num_features / 2), round(time_steps / 2)))
#             ax = sns.heatmap(sensitivity_array[np.newaxis, :], cmap="viridis", cbar_kws={'label': 'Sensitivity'}, square=True)
#             ax.set_title(f'Model Sensitivity Across Input Features for Head {i}')
#             ax.set_xlabel('Input Feature Group')
#             ax.set_ylabel('Head')

#             # Set the x-ticks to be the middle of each group of 10 features
#             ticks = np.arange(round(num_agents / 2), num_features, num_agents)  # start/stop/step
#             labels = hparams.feature_names
#             ax.set_xticks(ticks) # Apply custom ticks and labels
#             ax.set_xticklabels(labels, rotation=45)  # Rotate labels for better readability
#             # Drawing vertical lines to separate groups
#             for j in range(num_agents, num_features, num_agents):  # start/stop/step
#                 ax.axvline(x=j, color='white', linestyle='--', linewidth=1)  # Change color if needed
#             # Invert y-axis to have 0 at the bottom
#             ax.invert_yaxis()
#             # Save the plot
#             plt.savefig(f"{hparams.model_dir}saliency_{hparams.model_type}{hparams.window}_head_{i}.png")
#     else:
#         # Single-head case
#         # Average each batch's saliency map before stacking
#         # The initial shape of each saliency map in the list: (batch_size, output_size, batch_size, time_steps, features)
#         averaged_saliency = [tf.reduce_mean(batch_saliency, axis=[0, 2]) for batch_saliency in saliency]  # Output Shape: (O_size, T, F)
#         # Stack the averaged saliency maps
#         saliency_tensor = tf.stack(averaged_saliency)  # Output Shape: (B, O_size, T, F)
#         # Average over the batch dimension
#         mean_saliency = tf.reduce_mean(saliency_tensor, axis=0)  # Output shape: (O_size, T, F)
#         # Summarize sensitivity by computing the norm across the output dimension
#         sensitivity_summary = tf.norm(mean_saliency, axis=0)  # Shape: (T, F)
#         ## Sum of Absolute Values
#         # sum of the absolute values of the saliency map.
#         # This gives you a single number representing the total sensitivity of the model's outputs to all input features across all time steps.
#         total_sensitivity = tf.reduce_sum(tf.abs(sensitivity_summary))
#         print("Sum(abs(norm(mean(saliency: ", total_sensitivity)
#         sensitivity = tf.reduce_sum(tf.abs(mean_saliency))
#         print("Sum(abs(mean(saliency: ", sensitivity)
#         ##Norm
#         # Calculate a norm (e.g., L1, L2) across the entire saliency map.
#         # The L2 norm can be particularly useful as it represents the "energy" of the saliency map, providing a measure of overall model sensitivity.
#         overall_sensitivity = tf.norm(sensitivity_summary)
#         print("Norm(mean(saliency)): ", overall_sensitivity)
#         # Assume 'sensitivity_summary' is a TensorFlow 2D tensor of shape (time,#inputs)
#         # Convert it to a numpy array for plotting
#         sensitivity_array = sensitivity_summary.numpy()
#         # Find quantities to help with plotting
#         time_steps=sensitivity_array.shape[0]
#         num_features=sensitivity_array.shape[1]
#         num_features_per = len(hparams.feature_names)
#         num_agents=num_features//num_features_per
#         # Set up the plot dimensions and labels
#         plt.figure(figsize=(round(num_features/2),round(time_steps/2)))
#         ax = sns.heatmap(sensitivity_array, cmap="viridis", cbar_kws={'label': 'Sensitivity'}, square=True)
#         ax.set_title('Model Sensitivity Across Input Features')
#         ax.set_xlabel('Input Feature Group')
#         ax.set_ylabel('Time Step')
#         # Set the x-ticks to be the middle of each group of 10 features
#         ticks = np.arange(round(num_agents/2), num_features, num_agents)  # start/stop/step: Start at middle of #agents, increment by #agents
#         labels = hparams.feature_names
#         # Apply custom ticks and labels
#         ax.set_xticks(ticks)
#         ax.set_xticklabels(labels, rotation=45)  # Rotate labels for better readability
#         # Drawing vertical lines to separate groups
#         for i in range(num_agents, num_features, num_agents):  # start/stop/step: put white line between features
#             ax.axvline(x=i, color='white', linestyle='--', linewidth=1)  # Change color if needed
#         # Invert y-axis to have 0 at the bottom
#         ax.invert_yaxis()
#         # Show the plot
#         plt.savefig(f"{hparams.model_dir}saliency_{hparams.model_type}{hparams.window}.png")


## ORIGINAL (can't handle multihead output)
# def compute_saliency(model, dataset):
#     saliency_maps = []
#     for inputs, labels in dataset:
#         with tf.GradientTape() as tape:
#             tape.watch(inputs)
#             predictions = model(inputs)
#         # Derivative of outputs with respect to inputs
#         grads = tape.jacobian(predictions, inputs)
#         saliency_maps.append(grads)
#     return saliency_maps

# def print_saliency(hparams, model, test_dataset):

#     saliency = compute_saliency(model, test_dataset)

#     # ## ORIGINAL CODE (caused OOM error when STACKING)
#     # # Stack 24 lists of batch size 50 to create tensor
#     # # batch size appears repeated b/c input to output for each batch independently
#     # # B=batch, B_size=batch size (samples per batch), O=output, T=time, F=feature
#     # saliency_tensor = tf.stack(saliency)  # Shape: (B, B_size, O_size, B_size, T, F)
#     # # Average over the batch and sample dimensions
#     # mean_saliency = tf.reduce_mean(saliency_tensor, axis=[0, 1, 3])  # New shape: (O_size, B_size, T, F)
    
#     ## REDUCED MEMORY METHOD?
#     # Average each batch's saliency map before stacking
#     # The initial shape of each saliency map in the list: (batch_size, output_size, batch_size, time_steps, features)
#     averaged_saliency = [tf.reduce_mean(batch_saliency, axis=[0, 2]) for batch_saliency in saliency]  # Output Shape: (O_size, T, F)
#     # Stack the averaged saliency maps
#     saliency_tensor = tf.stack(averaged_saliency)  # Output Shape: (B, O_size, T, F)
#     # Average over the batch dimension
#     mean_saliency = tf.reduce_mean(saliency_tensor, axis=0)  # Output shape: (O_size, T, F)
#     # Summarize sensitivity by computing the norm across the output dimension
#     sensitivity_summary = tf.norm(mean_saliency, axis=0)  # Shape: (T, F)
#     ## Sum of Absolute Values
#     # sum of the absolute values of the saliency map.
#     # This gives you a single number representing the total sensitivity of the model's outputs to all input features across all time steps.
#     total_sensitivity = tf.reduce_sum(tf.abs(sensitivity_summary))
#     print("Sum(abs(norm(mean(saliency: ", total_sensitivity)
#     sensitivity = tf.reduce_sum(tf.abs(mean_saliency))
#     print("Sum(abs(mean(saliency: ", sensitivity)
#     ##Norm
#     # Calculate a norm (e.g., L1, L2) across the entire saliency map.
#     # The L2 norm can be particularly useful as it represents the "energy" of the saliency map, providing a measure of overall model sensitivity.
#     overall_sensitivity = tf.norm(sensitivity_summary)
#     print("Norm(mean(saliency)): ", overall_sensitivity)
#     # Assume 'sensitivity_summary' is a TensorFlow 2D tensor of shape (time,#inputs)
#     # Convert it to a numpy array for plotting
#     sensitivity_array = sensitivity_summary.numpy()
#     # Find quantities to help with plotting
#     time_steps=sensitivity_array.shape[0]
#     num_features=sensitivity_array.shape[1]
#     num_features_per = len(hparams.feature_names)
#     num_agents=num_features//num_features_per
#     # Set up the plot dimensions and labels
#     plt.figure(figsize=(round(num_features/2),round(time_steps/2)))
#     ax = sns.heatmap(sensitivity_array, cmap="viridis", cbar_kws={'label': 'Sensitivity'}, square=True)
#     ax.set_title('Model Sensitivity Across Input Features')
#     ax.set_xlabel('Input Feature Group')
#     ax.set_ylabel('Time Step')
#     # Set the x-ticks to be the middle of each group of 10 features
#     ticks = np.arange(round(num_agents/2), num_features, num_agents)  # start/stop/step: Start at middle of #agents, increment by #agents
#     labels = hparams.feature_names
#     # Apply custom ticks and labels
#     ax.set_xticks(ticks)
#     ax.set_xticklabels(labels, rotation=45)  # Rotate labels for better readability
#     # Drawing vertical lines to separate groups
#     for i in range(num_agents, num_features, num_agents):  # start/stop/step: put white line between features
#         ax.axvline(x=i, color='white', linestyle='--', linewidth=1)  # Change color if needed
#     # Invert y-axis to have 0 at the bottom
#     ax.invert_yaxis()
#     # Show the plot
#     plt.savefig(f"{hparams.model_dir}saliency_{hparams.model_type}{hparams.window}.png")


# def print_saliency(hparams, model, test_dataset):
#     # Compute Saliency
#     saliency = compute_saliency(model, test_dataset)

#     # Assume 'saliency' is a list of lists, where each inner list corresponds to a saliency map for a specific head.
#     averaged_saliency_heads = []
    
#     for head_saliency in saliency:
#         # Average each batch's saliency map before stacking
#         # The initial shape of each saliency map in the list: (batch_size, output_size, batch_size, time_steps, features)
#         averaged_saliency = [tf.reduce_mean(batch_saliency, axis=[0, 2]) for batch_saliency in head_saliency]  # Output Shape: (T, F)
#         # Stack the averaged saliency maps
#         saliency_tensor = tf.stack(averaged_saliency)  # Output Shape: (B, T, F)
#         # Average over the batch dimension
#         mean_saliency = tf.reduce_mean(saliency_tensor, axis=0)  # Output shape: (T, F)
#         averaged_saliency_heads.append(mean_saliency)
    
#     # Now 'averaged_saliency_heads' is a list of mean saliency maps for each head.
#     for i, mean_saliency in enumerate(averaged_saliency_heads):
#         # Summarize sensitivity by computing the norm across the output dimension
#         sensitivity_summary = tf.norm(mean_saliency, axis=0)  # Shape: (F)

#         ## Sum of Absolute Values
#         total_sensitivity = tf.reduce_sum(tf.abs(sensitivity_summary))
#         print(f"Sum(abs(norm(mean(saliency) for head {i}): ", total_sensitivity)
#         sensitivity = tf.reduce_sum(tf.abs(mean_saliency))
#         print(f"Sum(abs(mean(saliency) for head {i}): ", sensitivity)

#         ## Norm
#         overall_sensitivity = tf.norm(sensitivity_summary)
#         print(f"Norm(mean(saliency)) for head {i}: ", overall_sensitivity)

#         # Convert sensitivity summary to numpy array for plotting
#         sensitivity_array = sensitivity_summary.numpy()

#         # Find quantities to help with plotting
#         time_steps = sensitivity_array.shape[0]
#         num_features = sensitivity_array.shape[1]
#         num_features_per = len(hparams.feature_names)
#         num_agents = num_features // num_features_per

#         # Set up the plot dimensions and labels
#         plt.figure(figsize=(round(num_features / 2), round(time_steps / 2)))
#         ax = sns.heatmap(sensitivity_array, cmap="viridis", cbar_kws={'label': 'Sensitivity'}, square=True)
#         ax.set_title(f'Model Sensitivity Across Input Features for Head {i}')
#         ax.set_xlabel('Input Feature Group')
#         ax.set_ylabel('Time Step')
#         # Set the x-ticks to be the middle of each group of 10 features
#         ticks = np.arange(round(num_agents / 2), num_features, num_agents)  # start/stop/step
#         labels = hparams.feature_names
#         # Apply custom ticks and labels
#         ax.set_xticks(ticks)
#         ax.set_xticklabels(labels, rotation=45)  # Rotate labels for better readability
#         # Drawing vertical lines to separate groups
#         for j in range(num_agents, num_features, num_agents):  # start/stop/step
#             ax.axvline(x=j, color='white', linestyle='--', linewidth=1)  # Change color if needed
#         # Invert y-axis to have 0 at the bottom
#         ax.invert_yaxis()
#         # Save the plot
#         plt.savefig(f"{hparams.model_dir}saliency_{hparams.model_type}{hparams.window}_head_{i}.png")