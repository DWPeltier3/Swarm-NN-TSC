from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,multilabel_confusion_matrix,hamming_loss,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import os
import math
import torch
import random
from scipy.stats import gaussian_kde  # For KDE smoothing
# from utils.savevariables import append_to_csv



def outputs2tensor(outputs):
    # Initialize lists
    all_predictions, all_labels, all_inputs = [], [], []
    for batch_out in outputs:
        all_predictions.append(batch_out["predictions"].detach().cpu())
        all_labels.append(batch_out["labels"].detach().cpu())
        all_inputs.append(batch_out["inputs"].detach().cpu())

    # Concatenate the results
    test_predictions = torch.cat(all_predictions, dim=0)
    test_labels = torch.cat(all_labels, dim=0)
    test_inputs = torch.cat(all_inputs, dim=0)

    return test_predictions, test_labels, test_inputs


def calculate_ece(hparams, predictions, labels, num_bins=10, num_trials=10):
    """
    Calculate the Expected Calibration Error (ECE) for both DNN and BNN cases.
    
    For DNN:
      - 'predictions' is a NumPy array (or torch.Tensor) of shape [num_instances, num_classes]
      - The confidence for each instance is computed as the maximum softmax probability.
    
    For BNN:
      - The function loads the Monte Carlo (MC) predictions for each trial from disk.
      - For each trial, the mean class predictions (across MC samples) per instance is computed.
      - Then, ECE is calculated for each trial.
      - Finally, the 5th and 95th percentiles (CI) and the mean ECE across trials are returned.
    
    Parameters:
      hparams      : argparse.Namespace or similar, containing:
                     - bnn (Boolean): True if using a Bayesian NN.
                     - model_dir: directory for saving/loading results.
                     - mc_sample_directory: sub-directory with saved BNN MC predictions.
      predictions  : For DNN: predictions as a torch.Tensor or np.array.
                     For BNN: this parameter can be ignored since MC predictions are loaded from disk.
      labels       : Ground truth labels as a torch.Tensor or np.array.
      num_bins     : Number of bins to use when calculating ECE (default is 10; recommend 10-20).
      num_trials   : Number of MC trials to evaluate (only used for BNN; default is 10).
    """
    
    def ece_single(pred_mean, true_labels, num_bins):
        """
        Calculate ECE for a single set of predictions.
        
        Parameters:
            pred_mean   : Array of shape [num_instances, num_classes]
                          containing the softmax predictions (mean for BNN).
            true_labels : 1D array of true labels.
            num_bins    : Number of bins for calibration.
                          
        Returns:
            ece_value   : Computed Expected Calibration Error.
        """
        # Compute the predicted confidence and label for each instance
        confidences = np.max(pred_mean, axis=1)     # maximum probability per instance
        pred_labels = np.argmax(pred_mean, axis=1)  # predicted class
        num_instances = len(true_labels)
        ece = 0.0   # initial ECE (lower is better)
        
        # Create bin boundaries from 0 to 1
        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        
        # Iterate over each bin
        for i in range(num_bins):
            # Define bin range [bin_lower, bin_upper)
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i+1]
            # Find indices of instances whose confidence falls into the bin
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            bin_count = np.sum(in_bin)
            if bin_count > 0:
                # Average confidence and accuracy for instances in the bin
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(pred_labels[in_bin] == true_labels[in_bin])
                # Accumulate weighted difference
                ece += (bin_count / num_instances) * np.abs(avg_accuracy - avg_confidence)
        return ece

    # Ensure true_labels is a 1D NumPy array
    if hasattr(labels, "cpu"):
        true_labels = labels.squeeze(-1).cpu().numpy()
    else:
        true_labels = np.squeeze(labels)

    # DNN Case: Single prediction
    if not hparams.bnn:
        print(f"\n*** Calculating ECE ***")
        if hasattr(predictions, "cpu"):
            pred = predictions.cpu().numpy()
        else:
            pred = predictions
        ece_value = ece_single(pred, true_labels, num_bins)
        print(f"DNN ECE: {ece_value:.4f}")
        # return ece_value

    # BNN Case: Compute ECE for each trial and then the 5th/95th CI.
    else:
        ece_trials = []
        print(f"\n*** Calculating ECE ***")
        for trial in range(num_trials):
            # Construct path and load the MC predictions
            mc_path = os.path.join(hparams.model_dir, hparams.mc_sample_directory,
                                   f"saved_bnn_mc_predictions_trial_{trial}.npy")
            mc_predictions = np.load(mc_path)  # Shape: [MC_samples, num_instances, num_classes]
            # Compute mean prediction per instance (softmax)
            pred_mean = mc_predictions.mean(axis=0)  # Shape: [num_instances, num_classes]
            # Compute ECE for this trial
            trial_ece = ece_single(pred_mean, true_labels, num_bins)
            print(f"Trial {trial+1} ECE: {trial_ece:.4f}")
            ece_trials.append(trial_ece)
        
        ece_trials = np.array(ece_trials)
        mean_ece = np.mean(ece_trials)
        ci_lower = np.percentile(ece_trials, 5)
        ci_upper = np.percentile(ece_trials, 95)
        
        print(f"\nBNN ECE Mean: {mean_ece:.4f}")
        print(f"5th/95th CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        # return mean_ece, ci_lower, ci_upper


def perform_monte_carlo_sampling_trials(hparams, model, inputs, num_trials=10):
    """
    Perform Monte Carlo sampling multiple times for Bayesian Neural Network with
    Variational Inference.
    
    Parameters:
      hparams   : argparse.Namespace or similar, containing:
                    - num_monte_carlo: number of MC samples.
                    - model_dir: directory for saving MC results.
      model     : torch.nn.Module (the BNN model) with variational inference.
      inputs    : torch.Tensor containing the inputs (e.g., concatenated test set inputs).
      num_trials: Number of MC variational inference runs to complete; used for confidence intervals.
    """
    # ==========================================================
    # Load variables required
    # ==========================================================
    num_mc_samples = hparams.num_monte_carlo
    mc_directory = hparams.mc_sample_directory
    # Create the directory if it doesn't exist
    save_dir = os.path.join(hparams.model_dir, mc_directory)
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================================
    # Perform Variational Inference
    # ==========================================================
    print(f"\n*** PERFORMING MONTE CARLO SAMPLING ({num_mc_samples} samples) ***")
    model.eval()  # Ensure model is in eval mode
    for trial in range(num_trials):
        set_random_seed(trial)  # Set seed before each VI inference run
        print(f"*** TRIAL {trial + 1}/{num_trials} - PERFORMING VARIATIONAL INFERENCE ***")
        mc_samples = []
        with torch.no_grad():
            for _ in range(num_mc_samples):
                preds = model(inputs)  # VI-based stochastic forward pass
                mc_samples.append(preds)

        mc_samples = torch.stack(mc_samples).cpu().numpy()  # Shape: [MC_samples, num_instances, num_classes]
        # Save each trial's results with a distinct filename
        np.save(os.path.join(hparams.model_dir, mc_directory, f"saved_bnn_mc_predictions_trial_{trial}.npy"), mc_samples)
        # np.save(os.path.join(hparams.model_dir, mc_directory, f"saved_bnn_mc_predictions_mean_trial_{trial}.npy"), pred_mean)
    print("*** VARIATIONAL INFERENCE COMPLETED FOR ALL TRIALS ***")


def set_random_seed(seed):
    """Sets random seed for reproducibility in VI."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU consistency
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_uncertainty_trials(hparams, predictions, labels, num_trials=10):
    """
    Compute multiple uncertainty (variance) metrics.

    For a deterministic NN (DNN), `predictions` is assumed to be a torch.Tensor 
    (or NumPy array) of test outputs.

    For a Bayesian NN (BNN), this function will load the Monte Carlo (MC) results
    from disk and use them to compute additional metrics for multiple MC trials.
    
    Parameters:
      hparams      : argparse.Namespace or similar, containing:
                     - bnn (Boolean): True if using a Bayesian NN.
                     - model_dir: directory for saving results.
                     - output_type: e.g. 'mc' for multiclass.
                     - num_classes: used for normalizing uncertainty (if multiclass).
                     - num_attributes: used if multilabel.
                     - num_instances_visualize: number of instances to plot.
      predictions  : For DNN: test predictions (torch.Tensor or np.array).
                     For BNN: can be ignored (the function will load MC results).
      labels       : Ground truth labels (torch.Tensor or np.array).
      num_trials   : Number of times MC sampling (Variational Inference) was completed.
    
    Returns:
      None. Saves uncertainty metrics as CSV and calls plotting functions.
    """
    # ==========================================================
    # Define Class Names NN Trained on
    # ==========================================================
    class_names_pred = ["Greedy", "Greedy+", "Auction", "Auction+"] # class names NN trained on
    hparams.class_names_pred = class_names_pred

    # ==========================================================
    # Ensure true_labels are 1D
    # ==========================================================
    if isinstance(labels, torch.Tensor):
        true_labels = labels.squeeze(-1).cpu().numpy() # Ensure true_labels is 1D
    else:
        true_labels = np.squeeze(labels)

    # ==========================================================
    # Calculate uncertainty metrics
    # ==========================================================
    print(f"\n*** CALCULATING UNCERTAINTY ***")
    
    # BNN
    if hparams.bnn:
        mc_directory = hparams.mc_sample_directory
        uncertainty_directory = hparams.mc_uncertainty_directory
        # Create the directory if it doesn't exist
        save_dir = os.path.join(hparams.model_dir, uncertainty_directory)
        os.makedirs(save_dir, exist_ok=True)

        for trial in range(num_trials):
            
            print(f"\n*** BNN MC TRIAL {trial + 1}/{num_trials}")
            #### Load Monte Carlo predictions
            mc_path = os.path.join(hparams.model_dir, mc_directory, f"saved_bnn_mc_predictions_trial_{trial}.npy")
            mc_predictions = np.load(mc_path)           # Shape: [MC_samples, num_instances, num_classes]
            pred_mean = mc_predictions.mean(axis=0)     # Mean predictions per instance; Shape: [num_instances, num_classes]
            num_classes_pred = pred_mean.shape[-1]      # number of NN class predictions (based on NN output size)
            
            ##### Uncertainty Calculations

            # --- Variance and Standard Deviation ---
            # per instance; sum the variance across the class dimension.
            var = np.sum(np.var(mc_predictions, axis=0), axis=1)  # Shape: [num_instances]
            std = np.sqrt(var)
            # print(f"\nVariance mean: {var.mean():.4f}")
            # print(f"Standard deviation mean: {std.mean():.4f}")
            
            # --- Fischer Entropy ---
            # For multiclass:
            if hparams.output_type == 'mc':
                entropy_fischer = -np.sum(pred_mean * np.log(pred_mean + 1e-14), axis=1)
                norm_entropy_fischer = entropy_fischer / np.log2(num_classes_pred) # Normalized entropy for number of classes or attributes
            else:
                # For multilabel:
                entropy_fischer = -(pred_mean * np.log(pred_mean + 1e-14)) - ((1 - pred_mean) * np.log((1 - pred_mean) + 1e-14))
                norm_entropy_fischer = entropy_fischer / np.log2(2 ** hparams.num_attributes) # Normalized entropy for number of classes or attributes
            # print(f"\nFischer Entropy mean: {entropy_fischer.mean():.4f}")
            # print(f"Normalized Fischer Entropy mean: {norm_entropy_fischer.mean():.4f}")
            
            # --- Kwon Metrics ---
            epistemic_kwon = np.sum(np.mean(mc_predictions ** 2, axis=0) - np.mean(mc_predictions, axis=0) ** 2, axis=1)
            aleatoric_kwon = np.sum(np.mean(mc_predictions * (1 - mc_predictions), axis=0), axis=1)
            # print(f"\nEpistemic (Kwon) mean: {epistemic_kwon.mean():.4f}")
            # print(f"Aleatoric (Kwon) mean: {aleatoric_kwon.mean():.4f}")
            
            # --- Depeweg Metrics --- (use and reference this paper)
            if hparams.output_type == 'mc':
                aleatoric_depeweg = np.mean(-np.sum(mc_predictions * np.log(mc_predictions + 1e-14), axis=2), axis=0)
            else:
                aleatoric_depeweg = np.mean(-(mc_predictions * np.log(mc_predictions + 1e-14)) -
                                            (1 - mc_predictions) * np.log((1 - mc_predictions) + 1e-14), axis=0)
            epistemic_depeweg = entropy_fischer - aleatoric_depeweg
            # print(f"\nEpistemic (Depeweg) mean: {epistemic_depeweg.mean():.4f}")
            # print(f"Aleatoric (Depeweg) mean: {aleatoric_depeweg.mean():.4f}")
            
            # print(f"\n*** SAVING METRICS ***")
            ##### Save results for plotting
            y_pred = np.argmax(pred_mean, axis=-1) # predicted class (single integer=index or max prob) using Mean PDF across all MC runs; [num_test_samples]
            uncertainty_df = pd.DataFrame({
                'instance': np.arange(len(entropy_fischer)),
                'true_label': true_labels,
                'pred_label': y_pred,
                'variance': var,
                'standard_dev': std,
                'entropy_fischer': entropy_fischer,
                'norm_entropy_fischer': norm_entropy_fischer,
                'epistemic_kwon': epistemic_kwon,
                'aleatoric_kwon': aleatoric_kwon,
                'epistemic_depeweg': epistemic_depeweg,
                'aleatoric_depeweg': aleatoric_depeweg
            })
            uncertainty_df.to_csv(os.path.join(hparams.model_dir, uncertainty_directory, f"saved_bnn_testset_uncertainty_metrics_trial_{trial}.csv"), index=False)
            
            calculate_average_uncertainty(hparams, uncertainty_df, trial)
            uncertainty_metrics = ['entropy_fischer', 'epistemic_kwon', 'aleatoric_kwon', 'epistemic_depeweg', 'aleatoric_depeweg']
            for uncertainty_metric in uncertainty_metrics:
                sort_and_calc_vs_data_retained(hparams, trial, uncertainty_metric)
            
    else:
        print(f"\n*** DNN UNCERTAINTY ***")
        # For a deterministic NN (DNN)
        # Convert predictions to a NumPy array.
        if hasattr(predictions, "cpu"):
            pred = predictions.cpu().numpy()
        else:
            pred = predictions
        np.save(os.path.join(hparams.model_dir, "saved_dnn_predictions.npy"), pred)
        num_classes_pred = pred.shape[-1]      # number of NN class predictions (based on NN output size)
        
        ##### Uncertainty Calculations

        # --- Fischer Entropy ---
        if hparams.output_type == 'mc':
            entropy_fischer = -np.sum(pred * np.log(pred + 1e-14), axis=-1)
            norm_entropy_fischer = entropy_fischer / np.log2(num_classes_pred) # Normalized entropy for number of classes
        else:
            entropy_fischer = -(pred * np.log(pred + 1e-14)) - ((1 - pred) * np.log((1 - pred) + 1e-14))
            norm_entropy_fischer = entropy_fischer / np.log2(2 ** hparams.num_attributes) # Normalized entropy for number of attributes
        print(f"\nFischer Entropy mean: {entropy_fischer.mean():.4f}")
        print(f"Normalized Fischer Entropy mean: {norm_entropy_fischer.mean():.4f}")
        
        # --- Kwon (Aleatoric) ---
        aleatoric_kwon = np.sum(pred * (1 - pred), axis=-1)
        print(f"Aleatoric (Kwon) mean: {aleatoric_kwon.mean():.4f}")
        
        # print(f"\n*** DNN SAVING METRICS ***")
        ##### Save results for plotting
        y_pred = np.argmax(pred, axis=-1) # predicted class (single integer=index or max prob); [num_test_samples]
        uncertainty_df = pd.DataFrame({
            'instance': np.arange(len(entropy_fischer)),
            'true_label': true_labels,
            'pred_label': y_pred,
            'entropy_fischer': entropy_fischer,
            'norm_entropy_fischer': norm_entropy_fischer,
            'aleatoric_kwon': aleatoric_kwon
        })
        uncertainty_df.to_csv(os.path.join(hparams.model_dir, "saved_dnn_testset_uncertainty_metrics.csv"), index=False)
        
        calculate_average_uncertainty(hparams, uncertainty_df)
        uncertainty_metrics = ['entropy_fischer', 'aleatoric_kwon']
        for uncertainty_metric in uncertainty_metrics:
            sort_and_calc_vs_data_retained(hparams, uncertainty_metric=uncertainty_metric)


def calculate_average_uncertainty(hparams, uncertainty_df, trial=None):
    """
    Compute class average uncertainty metrics.
    
    Parameters:
      hparams      : argparse.Namespace or similar, containing:
                     - bnn (Boolean): True if using a Bayesian NN.
                     - model_dir: directory for saving results.
                     - num_classes: number of inference classes presented to NN during inference.
                     - class_names: class names presented to NN during inference.
      uncertainty_df: Pandas dataframe with uncertainty metrics.
      trial         : for BNN, provides Monte Carlo trial number.
    
    Returns:
      None. Saves average uncertainty metrics as CSV.
    """
    # ==========================================================
    # Load variables required
    # ==========================================================
    num_classes = hparams.num_classes                   # number of inference classes presented to NN during inference
    class_names = hparams.class_names                   # class names presented to NN during inference
    true_labels = uncertainty_df['true_label'].values   # Shape: [num_test_samples]
    y_pred = uncertainty_df['pred_label'].values        # Shape: [num_test_samples]

    # ==========================================================
    # Compute CLASS AVERAGES for every uncertainty metric
    # ==========================================================
    class_averages = []
    for cls in range(num_classes):
        indices = np.where(true_labels == cls)[0]   # Finds all instances where the true label matches the current class; [0] ensures 1D
        if len(indices) == 0:
            print(f"No test instances found for class {class_names[cls]}")
            continue
        # Save the class averages into a dictionary
        avg_dict = {}
        avg_dict['class'] = hparams.class_names[cls] if hasattr(hparams, 'class_names') else str(cls) # Use provided class names if available; otherwise, use the integer.
        avg_dict['num_instances'] = len(indices)
        avg_dict['avg_entropy_fischer'] = uncertainty_df.loc[indices, 'entropy_fischer'].mean()
        avg_dict['avg_norm_entropy_fischer'] = uncertainty_df.loc[indices, 'norm_entropy_fischer'].mean()
        if hparams.bnn:
            avg_dict['avg_variance'] = uncertainty_df.loc[indices, 'variance'].mean()
            avg_dict['avg_standard_dev'] = uncertainty_df.loc[indices, 'standard_dev'].mean()
            avg_dict['avg_epistemic_kwon'] = uncertainty_df.loc[indices, 'epistemic_kwon'].mean()
            avg_dict['avg_aleatoric_kwon'] = uncertainty_df.loc[indices, 'aleatoric_kwon'].mean()
            avg_dict['avg_epistemic_depeweg'] = uncertainty_df.loc[indices, 'epistemic_depeweg'].mean()
            avg_dict['avg_aleatoric_depeweg'] = uncertainty_df.loc[indices, 'aleatoric_depeweg'].mean()
        else:
            avg_dict['avg_aleatoric_kwon'] = uncertainty_df.loc[indices, 'aleatoric_kwon'].mean()
        # Also compute class accuracy
        avg_dict['avg_accuracy'] = np.mean(y_pred[indices] == cls)
        class_averages.append(avg_dict)
    
    # ==========================================================
    # Save/Print Results
    # ==========================================================
    class_avg_df = pd.DataFrame(class_averages)

    if hparams.bnn:
        file_suffix = "bnn"
        uncertainty_directory = hparams.mc_uncertainty_directory
        class_avg_path = os.path.join(
            hparams.model_dir,
            uncertainty_directory,
            f"saved_{file_suffix}_testset_class_average_uncertainty_trial_{trial}.csv"
            )
    else:
        file_suffix = "dnn"
        class_avg_path = os.path.join(hparams.model_dir, f"saved_{file_suffix}_testset_class_average_uncertainty.csv")
    class_avg_df.to_csv(class_avg_path, index=False)
    
    print("\n*** CLASS AVERAGES ***")
    print(class_avg_df.to_string(index=False))


def sort_and_calc_vs_data_retained(hparams, trial=1, uncertainty_metric='entropy_fischer'):
    """
    Sorts and calculates metrics vs. percentage of test instances retained, 
    based on the specified uncertainty metric.

    Parameters:
        hparams             : argparse.Namespace or similar, containing:
                                - bnn (Boolean): True if using a Bayesian NN.
                                - model_dir: directory for saving results.
                                - uncertainty_directory: where to save results
                                - num_classes: number of inference classes presented to NN during inference.
                                - class_names: class names presented to NN during inference.
        uncertainty_metric  : The column name of the uncertainty metric to use (default: 'entropy_fischer').
        trial               : for BNN, provides Monte Carlo trial number.
    """
    # ==========================================================
    # Load variables required
    # ==========================================================
    num_classes = hparams.num_classes                   # number of inference classes presented to NN during inference
    class_names = hparams.class_names                   # class names presented to NN during inference
    
    # Determine the correct file paths
    if hparams.bnn:
        file_suffix = "bnn"
        uncertainty_directory = hparams.mc_uncertainty_directory
        uncertainty_path = os.path.join(
            hparams.model_dir,
            uncertainty_directory,
            f"saved_{file_suffix}_testset_uncertainty_metrics_trial_{trial}.csv"
            )
        sorted_uncertainty_path = os.path.join(
            hparams.model_dir,
            uncertainty_directory,
            f"saved_{file_suffix}_testset_uncertainty_metrics_{uncertainty_metric}_sorted_trial_{trial}.csv"
            )
    else:
        file_suffix = "dnn"
        uncertainty_path = os.path.join(
            hparams.model_dir,
            f"saved_{file_suffix}_testset_uncertainty_metrics.csv"
            )
        sorted_uncertainty_path = os.path.join(
            hparams.model_dir,
            f"saved_{file_suffix}_testset_uncertainty_metrics_{uncertainty_metric}_sorted.csv"
            )
    # Load uncertainty metrics
    if not os.path.exists(uncertainty_path):
        print(f"[WARNING] Accuracy plot skipped: {uncertainty_path} not found.")
        return
    uncertainty_df = pd.read_csv(uncertainty_path)
    # Check if the specified uncertainty metric exists
    if uncertainty_metric not in uncertainty_df.columns:
        raise ValueError(f"Uncertainty metric '{uncertainty_metric}' not found in the dataset. "
                         f"Available metrics: {list(uncertainty_df.columns)}")

    # ==========================================================
    # Sort and Calculate Metrics
    # ==========================================================
    ## Sort instances by the specified uncertainty metric (descending = highest uncertainty first)
    sorted_uncertainty_df = uncertainty_df.sort_values(by=uncertainty_metric, ascending=False).reset_index(drop=True)

    # Extract sorted_df values as required for calculations or saving
    uncertainty = sorted_uncertainty_df[uncertainty_metric].values  # Extracted so column title tells which uncertainty used
    true_labels = sorted_uncertainty_df['true_label'].values
    predicted_labels = sorted_uncertainty_df['pred_label'].values
    instances = sorted_uncertainty_df["instance"].values
    num_instances = len(sorted_uncertainty_df)

    # Calculate RNG SEED for each instance
    startseed = 7501                                    # RNG seed starting number
    num_per_class = num_instances // num_classes        # Assuming equal distribution of classes
    class_instance_numbers = instances % num_per_class  # Compute class instance number (i.e. between 0 and num_per_class)
    seeds = startseed + class_instance_numbers          # Compute seed based on class instance number

    # init lists for results
    data_retained_percent, accuracy_percent = [], []
    class_retention = {cls: [] for cls in range(num_classes)}

    percentage_to_keep = 0.5  # 50% of instances
    num_to_keep = int(num_instances * percentage_to_keep)

    for i in range(num_to_keep):
        # First iteration includes all instances
        retained_true_labels = true_labels[i:]
        retained_pred_labels = predicted_labels[i:]
        # Compute accuracy
        accuracy = np.mean(retained_true_labels == retained_pred_labels) * 100
        # Compute percentage of each class retained
        for cls in range(num_classes):
            class_retention[cls].append((np.sum(retained_true_labels == cls) / num_per_class) * 100)
        # Store values
        data_retained_percent.append(((num_instances - i) / num_instances) * 100)
        accuracy_percent.append(accuracy)
    # Ensure all columns have same length (match the size of "for loop" above)
    uncertainty = uncertainty[:num_to_keep]
    true_label = true_labels[:num_to_keep]
    true_label_class = [class_names[label] for label in true_label]  # convert true label to class name
    instances = instances[:num_to_keep]
    seeds = seeds[:num_to_keep]

    # ==========================================================
    # Create and Save sorted .csv
    # ==========================================================
    sorted_df = pd.DataFrame({
        uncertainty_metric: uncertainty,
        "true_label": true_label,
        "class": true_label_class,
        "instance": instances,
        "seed": seeds,
        "accuracy_percent": accuracy_percent,
        "data_retained_percent": data_retained_percent
        })
    # Add class retention percentages to the DataFrame
    for cls in range(num_classes):
        class_retention[cls] = np.array(class_retention[cls]) # Convert class retention lists to numpy arrays for consistency (not required for plotting)
        class_name = class_names[cls]  # Get the class name
        sorted_df[f"{class_name}_retained_percent"] = class_retention[cls]  # Use class name in column name
    # Save sorted .csv
    sorted_df.to_csv(sorted_uncertainty_path, index=False)


def show_uncertainty_trials(hparams, num_trials=10, ood_classes=None):
    """
    Plot uncertainty metrics.
    
    Parameters:
      hparams      : argparse.Namespace or similar, containing:
                     - bnn (Boolean): True if using a Bayesian NN.
                     - num_instances_visualize: number of instances to plot.
      num_trials   : Number of Monte Carlo samplings conducted
      ood_classes  : Out of Distribution class label integers (ie. [8,9]).
    
    Returns:
      None. Calls plotting functions.
    """

    # ==========================================================
    # Make Plots
    # ==========================================================
    print(f"\n*** PLOTTING ***")

    plot_uncertainty_histograms(hparams)

    if hparams.bnn:
        plot_IDODD_histograms(hparams, num_trials, uncertainty_metric='epistemic_depeweg', selected_ood_classes=ood_classes)
        plot_IDODD_KDE(hparams, num_trials, uncertainty_metric='epistemic_depeweg', selected_ood_classes=ood_classes)
        plot_bnn_bar_plot(hparams, num_trials, uncertainty_metric="depeweg")    # "depeweg" or "entropy"
        plot_bnn_bar_plot(hparams, num_trials, uncertainty_metric="entropy")    # "depeweg" or "entropy"
        uncertainty_metrics = ['entropy_fischer', 'epistemic_kwon', 'aleatoric_kwon', 'epistemic_depeweg', 'aleatoric_depeweg']
        for uncertainty_metric in uncertainty_metrics:
            calculate_and_plot_avu(hparams, uncertainty_metric=uncertainty_metric, num_trials=num_trials, tau_steps=101)
            plot_vs_data_retained(hparams, num_trials, uncertainty_metric)
    
    else:
        plot_dnn_bar_plot(hparams)
        uncertainty_metrics = ['entropy_fischer', 'aleatoric_kwon']
        for uncertainty_metric in uncertainty_metrics:
            calculate_and_plot_avu(hparams, uncertainty_metric=uncertainty_metric, num_trials=num_trials, tau_steps=101)
            plot_vs_data_retained(hparams, uncertainty_metric=uncertainty_metric)
    
    plot_avu_all_metrics_aggregated(hparams, uncertainty_metrics)
    plot_avu_all_metrics(hparams, uncertainty_metrics)
    plot_vs_data_retained_all_metrics_aggregated(hparams, uncertainty_metrics)
    plot_vs_data_retained_all_metrics(hparams, uncertainty_metrics)

    for instance in range(hparams.num_instances_visualize):
        plot_class_probabilities(hparams, instance_number=instance)


def calculate_and_plot_avu(hparams, uncertainty_metric='entropy_fischer', num_trials=10, tau_steps=101):
    """
    Calculates AvU (Accuracy vs. Uncertainty) across a range of thresholds (τ in [0,1])
    for a given uncertainty metric. For each trial, the function loads the corresponding
    uncertainty CSV file and computes, for each τ:
      - AC: Accurate and Certain
      - AU: Accurate and Uncertain
      - IC: Inaccurate and Certain
      - IU: Inaccurate and Uncertain
    Then AvU is computed as:
    
        AvU = (AC + IU) / (AC + AU + IC + IU)
    
    This function then saves a CSV file containing, for each τ, the computed counts and
    AvU for each trial, along with aggregated statistics (mean, 5th and 95th percentiles) across trials.
    Finally, it plots AvU (mean with 5/95 CI) vs. τ.
    
    Parameters:
        hparams: Object containing:
            - model_dir: Base directory for files.
            - mc_uncertainty_directory: (if BNN) Sub-directory where CSV files are stored.
            - bnn: Boolean flag; if True, multiple trials are assumed.
        uncertainty_metric: String; the uncertainty metric column name (e.g., 'entropy_fischer').
        num_trials: Number of trials (only used if hparams.bnn is True). For DNN, only a single file is used.
        tau_steps: Number of τ values between 0 and 1 (inclusive) to evaluate.
    """

    # Define the τ values as a numpy array.
    tau_values = np.linspace(0, 1, tau_steps)
    
    # Determine file suffix and base directory depending on the model type.
    if hparams.bnn:
        file_suffix = "bnn"
        base_dir = os.path.join(hparams.model_dir, hparams.mc_uncertainty_directory)
        trials = list(range(num_trials))
    else:
        file_suffix = "dnn"
        base_dir = hparams.model_dir
        trials = [0]  # Only one trial for DNN
    
    # Initialize dictionaries to store results for each trial.
    # For each trial, we store arrays (one per τ) for AC, AU, IC, IU, and AvU.
    results = {}
    for trial in trials:
        results[trial] = {
            "AC": np.zeros(tau_steps, dtype=int),
            "AU": np.zeros(tau_steps, dtype=int),
            "IC": np.zeros(tau_steps, dtype=int),
            "IU": np.zeros(tau_steps, dtype=int),
            "AvU": np.zeros(tau_steps, dtype=float)
        }
    
    # Loop over each trial.
    for trial in trials:
        # Build the CSV file path.
        if hparams.bnn:
            csv_path = os.path.join(
                base_dir,
                f"saved_{file_suffix}_testset_uncertainty_metrics_trial_{trial}.csv"
            )
        else:
            csv_path = os.path.join(
                base_dir,
                f"saved_{file_suffix}_testset_uncertainty_metrics.csv"
            )
        
        if not os.path.exists(csv_path):
            print(f"[WARNING] CSV file not found: {csv_path}. Skipping trial {trial}.")
            continue
        
        # Load the CSV data.
        df = pd.read_csv(csv_path)
        # Ensure required columns exist.
        if not set(["instance", "true_label", "pred_label", uncertainty_metric]).issubset(df.columns):
            print(f"[WARNING] Required columns missing in {csv_path}.")
            continue
        
        # Determine correctness: accurate if predicted equals true label.
        accurate = (df["true_label"].values == df["pred_label"].values)
        # Get the uncertainty values from the specified uncertainty_metric column.
        uncertainty_vals = df[uncertainty_metric].values
        
        ### *******************************
        # IF WANT NORMALIZED UNCERTAINTIES
            # better to compare different uncertainty ranges (tau = min to max)
        min_val = np.min(uncertainty_vals)
        max_val = np.max(uncertainty_vals)
        uncertainty_vals = (uncertainty_vals - min_val) / (max_val - min_val)
        
        # Loop over each τ value.
        for idx, tau in enumerate(tau_values):
            # For each instance, determine "certainty" based on tau.
            # Here, we consider an instance as "certain" if its uncertainty is below tau.
            is_certain = uncertainty_vals < tau
            is_uncertain = ~is_certain  # logical negation
            
            # Count outcomes:
            # AC: accurate and certain
            AC = np.sum(accurate & is_certain)
            # AU: accurate and uncertain
            AU = np.sum(accurate & is_uncertain)
            # IC: inaccurate and certain
            IC = np.sum(np.logical_not(accurate) & is_certain)
            # IU: inaccurate and uncertain
            IU = np.sum(np.logical_not(accurate) & is_uncertain)
            
            # Save counts.
            results[trial]["AC"][idx] = AC
            results[trial]["AU"][idx] = AU
            results[trial]["IC"][idx] = IC
            results[trial]["IU"][idx] = IU
            
            # Compute AvU.
            total = AC + AU + IC + IU
            if total > 0:
                results[trial]["AvU"][idx] = (AC + IU) / total
            else:
                results[trial]["AvU"][idx] = np.nan  # safeguard
            
    # Build a DataFrame to save the results.
    # The CSV will have one row per τ value.
    df_out = pd.DataFrame({"Tau": tau_values})
    
    # Append columns for each trial.
    for trial in trials:
        df_out[f"AC_trial{trial}"] = results[trial]["AC"]
        df_out[f"AU_trial{trial}"] = results[trial]["AU"]
        df_out[f"IC_trial{trial}"] = results[trial]["IC"]
        df_out[f"IU_trial{trial}"] = results[trial]["IU"]
        df_out[f"AvU_trial{trial}"] = results[trial]["AvU"]
    
    # If more than one trial, compute aggregated AvU statistics across trials.
    if len(trials) > 1:
        # Collect AvU arrays from each trial (each is of shape (tau_steps,))
        avu_trials = np.array([results[trial]["AvU"] for trial in trials])  # shape: (num_trials, tau_steps)
        agg_avu_mean = np.mean(avu_trials, axis=0)
        agg_avu_lower = np.percentile(avu_trials, 5, axis=0)
        agg_avu_upper = np.percentile(avu_trials, 95, axis=0)
        
        df_out["AvU_mean"] = agg_avu_mean
        df_out["AvU_lower"] = agg_avu_lower
        df_out["AvU_upper"] = agg_avu_upper
    else:
        # For single trial, aggregate columns are just the trial's values.
        df_out["AvU_mean"] = df_out["AvU_trial0"]
        df_out["AvU_lower"] = df_out["AvU_trial0"]
        df_out["AvU_upper"] = df_out["AvU_trial0"]
    
    # Determine the CSV save directory.
    if hparams.bnn:
        csv_save_dir = os.path.join(hparams.model_dir, hparams.mc_uncertainty_directory)
    else:
        csv_save_dir = hparams.model_dir
    
    # Save the results CSV.
    csv_save_path = os.path.join(csv_save_dir, f"saved_{file_suffix}_testset_AvU_using_{uncertainty_metric}.csv")
    df_out.to_csv(csv_save_path, index=False)
    
    # ==========================================================
    # Plotting: Plot AvU (mean and 5/95 CI) vs. Tau
    # ==========================================================
    plt.figure(figsize=(8, 6))
    plt.plot(df_out["Tau"], df_out["AvU_mean"], linestyle='-', linewidth=2, label='AvU (mean)')
    plt.fill_between(df_out["Tau"], df_out["AvU_lower"], df_out["AvU_upper"], alpha=0.3, label='5/95 CI')
    plt.xlabel("Tau (Normalized Uncertainty Threshold)\nInstances Certain if Uncertainty < Tau")
    plt.ylabel("AvU")
    plt.title(f"AvU vs. Tau (using {uncertainty_metric})")
    plt.legend()
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.gca().xaxis.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='black')
    plt.gca().xaxis.grid(visible=True, which='minor', linestyle=':', linewidth=0.25)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='black')
    plt.gca().yaxis.grid(visible=True, which='minor', linestyle=':', linewidth=0.25)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Invert x-axis if desired (optional; comment out if not needed)
    # plt.gca().invert_xaxis()
    
    # Save the figure.
    fig_save_path = os.path.join(hparams.model_dir, f"{file_suffix}_AvU_vs_tau_{uncertainty_metric}.png")
    plt.savefig(fig_save_path)
    plt.show()
    plt.close()

def plot_avu_all_metrics(hparams, uncertainty_metrics):
    """
    Plots AvU vs. Tau curves (AvU mean with 5/95 CI) for each uncertainty metric on one plot.
    
    For each uncertainty metric, it loads the corresponding CSV file using the naming convention:
        saved_{file_suffix}_testset_AvU_using_{metric}.csv
    where file_suffix is either "bnn" or "dnn" based on hparams.
    
    The CSV is assumed to contain the following columns:
        - "Tau": Normalized uncertainty thresholds (from 0 to 1)
        - "AvU_mean": Mean AvU values (aggregated if multiple trials)
        - "AvU_lower": 5th percentile of AvU values
        - "AvU_upper": 95th percentile of AvU values
    
    The function plots each metric's AvU mean curve with a fill representing the 5/95 CI, adds a legend,
    and saves the resulting plot to the model directory.
    
    Parameters:
        hparams: Object with attributes:
            - model_dir: Base directory.
            - mc_uncertainty_directory: For BNN, the subdirectory where CSV files are stored.
            - bnn: Boolean flag indicating if BNN is used.
        uncertainty_metrics: List of uncertainty metric strings.
    """

    # Determine base directory and file suffix
    if hparams.bnn:
        base_dir = os.path.join(hparams.model_dir, hparams.mc_uncertainty_directory)
        file_suffix = "bnn"
    else:
        base_dir = hparams.model_dir
        file_suffix = "dnn"

    # Initialize the plot
    plt.figure(figsize=(8, 6))
    
    # Set up a color map (one distinct color per metric)
    cmap = plt.get_cmap("tab10")
    
    # Loop over each uncertainty metric.
    for idx, metric in enumerate(uncertainty_metrics):
        csv_path = os.path.join(base_dir, f"saved_{file_suffix}_testset_AvU_using_{metric}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] CSV file not found: {csv_path}. Skipping metric '{metric}'.")
            continue
        
        df = pd.read_csv(csv_path)
        if not set(["Tau", "AvU_mean", "AvU_lower", "AvU_upper"]).issubset(df.columns):
            print(f"[WARNING] Required columns missing in {csv_path}. Skipping metric '{metric}'.")
            continue
        
        tau = df["Tau"].values
        avu_mean = df["AvU_mean"].values
        avu_lower = df["AvU_lower"].values
        avu_upper = df["AvU_upper"].values
        
        # Choose a color from the colormap
        color = cmap(idx % 10)
        
        # Plot the AvU mean curve with 5/95 CI fill
        plt.plot(tau, avu_mean, linestyle='-', linewidth=2, label=metric, color=color)
        plt.fill_between(tau, avu_lower, avu_upper, color=color, alpha=0.3)
    
    plt.xlabel("Tau (Normalized Uncertainty Threshold)\nInstances Certain if Uncertainty < Tau")
    plt.ylabel("AvU")
    plt.title("AvU vs. Tau (All Uncertainty Metrics)")
    plt.legend()
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.gca().xaxis.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='black')
    plt.gca().xaxis.grid(visible=True, which='minor', linestyle=':', linewidth=0.25)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='black')
    plt.gca().yaxis.grid(visible=True, which='minor', linestyle=':', linewidth=0.25)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the figure to the model directory.
    save_path = os.path.join(hparams.model_dir, f"{file_suffix}_AvU_vs_tau_0_all_metrics.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_avu_all_metrics_aggregated(hparams, uncertainty_metrics):
    """
    Loads the AvU CSV files for each (Normalized) uncertainty metric, aggregates the AvU curves,
    calculates the mean and 5/95 CI for AvU across metrics, saves the aggregated statistics,
    and plots AvU (mean and CI) vs. Tau.

    CSV file naming:
        Loaded: saved_{file_suffix}_testset_AvU_using_{metric}.csv
        Saved:  saved_{file_suffix}_testset_AvU_statistics_across_all_metrics.csv

    Parameters:
        hparams: Object with attributes:
            - model_dir: Base directory.
            - mc_uncertainty_directory: For BNN, the subdirectory where CSV files are stored.
            - bnn: Boolean flag indicating if BNN is used.
        uncertainty_metrics: List of uncertainty metric strings.
    """

    # Determine base directory and file suffix
    if hparams.bnn:
        base_dir = os.path.join(hparams.model_dir, hparams.mc_uncertainty_directory)
        file_suffix = "bnn"
    else:
        base_dir = hparams.model_dir
        file_suffix = "dnn"

    # Load sample CSV to get common Tau axis.
    sample_csv = os.path.join(base_dir, f"saved_{file_suffix}_testset_AvU_using_{uncertainty_metrics[0]}.csv")
    sample_df = pd.read_csv(sample_csv)
    tau_values = sample_df["Tau"].values  # Normalized "Tau" as the common x-axis

    # Initialize list to store AvU curves for each metric.
    avu_curves_list = []
    
    # Loop over each uncertainty metric.
    for metric in uncertainty_metrics:
        csv_path = os.path.join(base_dir, f"saved_{file_suffix}_testset_AvU_using_{metric}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] CSV file not found: {csv_path}. Skipping metric {metric}.")
            continue
        df = pd.read_csv(csv_path)
        # Use aggregated column if available; otherwise, use the single-trial column.
        if "AvU_mean" in df.columns:
            avu_curve = df["AvU_mean"].values
        # else:
        #     avu_curve = df["AvU_trial0"].values
        avu_curves_list.append(avu_curve)
    
    if len(avu_curves_list) == 0:
        print("No valid AvU curves loaded. Exiting function.")
        return

    # Convert list to numpy array: shape (num_metrics, num_points)
    avu_array = np.array(avu_curves_list)
    
    # Calculate aggregated statistics across metrics.
    agg_avu_mean = np.mean(avu_array, axis=0)
    agg_avu_lower = np.percentile(avu_array, 5, axis=0)
    agg_avu_upper = np.percentile(avu_array, 95, axis=0)
    
    # ==========================================================
    # Plotting: Plot aggregated AvU (mean and 5/95 CI) vs. Tau.
    # ==========================================================
    plt.figure(figsize=(8, 6))
    plt.plot(tau_values, agg_avu_mean, linestyle='-', linewidth=2, label='AvU (mean)')
    plt.fill_between(tau_values, agg_avu_lower, agg_avu_upper, alpha=0.3, label='5/95 CI')
    plt.xlabel("Tau (Normalized Uncertainty Threshold)\nInstances Certain if Uncertainty < Tau")
    plt.ylabel("AvU")
    plt.title("AvU vs. Tau (All Metrics Mean & 5/95 CI)")
    plt.legend()
    plt.gca().set_xlim(0, 1)
    plt.gca().set_ylim(0, 1)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.gca().xaxis.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='black')
    plt.gca().xaxis.grid(visible=True, which='minor', linestyle=':', linewidth=0.25)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='black')
    plt.gca().yaxis.grid(visible=True, which='minor', linestyle=':', linewidth=0.25)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the plot.
    plot_save_path = os.path.join(hparams.model_dir, f"{file_suffix}_AvU_vs_tau_0_all_metrics_aggregated.png")
    plt.savefig(plot_save_path)
    plt.show()
    plt.close()
    
    # ==========================================================
    # Saving Aggregated AvU Statistics as CSV
    # ==========================================================
    stats_df = pd.DataFrame({
        "Tau": tau_values,
        "AvU_mean": agg_avu_mean,
        "AvU_lower": agg_avu_lower,
        "AvU_upper": agg_avu_upper
    })
    
    csv_save_path = os.path.join(base_dir, f"saved_{file_suffix}_testset_AvU_statistics_across_all_metrics.csv")
    stats_df.to_csv(csv_save_path, index=False)


def plot_uncertainty_histograms(hparams):
    """Plots histograms of uncertainty metrics for DNN or BNN models."""
    # ==========================================================
    # Load variables required
    # ==========================================================
    # Determine the correct file paths
    if hparams.bnn:
        file_suffix = "bnn"
        uncertainty_directory = hparams.mc_uncertainty_directory
        trial = 0       # arbitrarily set to first MC trial
        uncertainty_path = os.path.join(
            hparams.model_dir,
            uncertainty_directory,
            f"saved_bnn_testset_uncertainty_metrics_trial_{trial}.csv"
            )
    else:
        file_suffix = "dnn"
        uncertainty_path = os.path.join(hparams.model_dir, f"saved_{file_suffix}_testset_uncertainty_metrics.csv")
    # Ensure the file exists
    if not os.path.exists(uncertainty_path):
        print(f"[WARNING] Uncertainty metrics file not found: {uncertainty_path}")
        return
    # Load uncertainty metrics
    uncertainty_df = pd.read_csv(uncertainty_path)
    
    # Define uncertainty types and corresponding columns
    entropy_columns = ['entropy_fischer', 'norm_entropy_fischer']
    aleatoric_columns = ['aleatoric_kwon', 'aleatoric_depeweg'] if hparams.bnn else ['aleatoric_kwon']
    epistemic_columns = ['epistemic_kwon', 'epistemic_depeweg'] if hparams.bnn else []

    # Combine all columns and titles for the 3x2 layout
    all_columns = entropy_columns + aleatoric_columns + epistemic_columns
    all_titles = (
        [f"Entropyset : {col}" for col in entropy_columns] +
        [f"Aleatoric: {col}" for col in aleatoric_columns] +
        [f"Epistemic: {col}" for col in epistemic_columns]
    )
    
    # Define a single color for each row
    row_colors = ['green', 'blue', 'orange']  # Entropy, Aleatoric, Epistemic
    all_colors = row_colors[0:1] * 2 + row_colors[1:2] * 2 + row_colors[2:3] * 2  # Repeat colors for each row

    # Create a grid of subplots
    num_rows = (len(all_columns) + 1) // 2  # If odd, one row will have 1 plot
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, num_rows * 4), sharex=False)
    # fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
    axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration

    # Plot each histogram
    for i, (col, title, color) in enumerate(zip(all_columns, all_titles, all_colors)):
        
        if col not in uncertainty_df.columns:
            print(f"[WARNING] {col} not found in dataset. Skipping plot.")
            continue
        
        # Plot histogram
        axes[i].hist(uncertainty_df[col], bins=50, alpha=0.7, label=col, color=color)

        # Calculate mean and add dashed red line
        mean_value = uncertainty_df[col].mean()
        axes[i].axvline(mean_value, color='red', linestyle='--', linewidth=1.5)
        
        # Label the mean horizontally, slightly right of the dashed line
        ymax = axes[i].get_ylim()[1] / 2  # Halfway up the y-axis
        axes[i].text(mean_value + 0.01 * axes[i].get_xlim()[1], ymax,  # Slightly offset to the right
                     f"mean = {mean_value:.2f}", color='red', fontsize=10, ha='left', va='center')

        # Set titles and labels
        axes[i].set_title(title)
        axes[i].set_xlabel("Uncertainty")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()
        axes[i].grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    # plt.savefig(os.path.join(hparams.model_dir, "BNN_uncertainty_histograms.png"))
    plt.savefig(os.path.join(hparams.model_dir, f"{file_suffix}_uncertainty_histograms.png"))
    plt.show()
    plt.close()


def plot_IDODD_histograms(hparams, num_trials=10, uncertainty_metric='epistemic_depeweg', selected_ood_classes=None):
    """
    Plots density histograms for the uncertainty metric for in-distribution (ID) and 
    out-of-distribution (OOD) classes on the same plot. For each group, the 5th and 
    95th percentiles (i.e., the 5/95 CI) are shown as vertical lines.
    
    Parameters:
        hparams: Hyperparameter object that should include:
            - model_dir: Base directory for model results.
            - num_classes: Total number of classes.
            - class_names: List of class names.
            - uncertainty_directory: Directory name where uncertainty CSVs are stored.
              (Ensure hparams contains the proper attribute.)
        num_trials: Number of trials to aggregate over.
        uncertainty_metric: The column name in the CSV to use for uncertainty (default is 'epistemic_depeweg').
        selected_ood_classes: List of class indices to include in the OOD histogram.
                              If None, defaults to all classes with index 4 and above.
    
    CSV file format:
        - Each CSV file is named "saved_bnn_testset_uncertainty_metrics_trial_{trial}.csv".
        - The CSV has a column named "instance" (used to identify each instance) and one
          column corresponding to the given uncertainty_metric.
        - All instances for each class are stored sequentially, and we assume equal distribution
          of instances among classes.
        - The first 4 classes (i.e., classes with index 0-3) are considered in-distribution (ID)
          and the remaining classes are considered out-of-distribution (OOD).
    
    Mathematical Formulation:
    
    Given a set of uncertainty values for a group (ID or OOD) denoted by:
    
        U = {u_1, u_2, ..., u_N}
    
    The 5th percentile (p₅) and 95th percentile (p₉₅) are defined as:
    
        p₅ = Percentile₅(U)    and    p₉₅ = Percentile₉₅(U)
    
    These percentiles are computed such that:
    
        p₅ is the smallest value u satisfying:
            (1/N) * ∑_{i=1}^{N} I(u_i ≤ u) ≥ 0.05
        p₉₅ is the smallest value u satisfying:
            (1/N) * ∑_{i=1}^{N} I(u_i ≤ u) ≥ 0.95
    where I is the indicator function.
    
    The vertical lines drawn at these percentiles on the histogram help visualize the spread of the data.
    """
    
    num_classes = hparams.num_classes
    uncertainty_directory = hparams.mc_uncertainty_directory  # Ensure this attribute exists in hparams

    # Default: If no specific OOD classes provided, include all OOD classes (index ≥ 4)
    if selected_ood_classes is None:
        selected_ood_classes = list(range(4, num_classes))

    # Lists to hold uncertainty values across all trials for ID and OOD classes.
    uncertainties_ID = []
    uncertainties_OOD = []

    # Loop over each trial and aggregate uncertainty values.
    for trial in range(num_trials):
        file_path = os.path.join(
            hparams.model_dir,
            uncertainty_directory,
            f"saved_bnn_testset_uncertainty_metrics_trial_{trial}.csv"
        )
        df = pd.read_csv(file_path)
        
        # Determine the number of instances per class (assumes equal distribution).
        num_instances = len(df)
        num_per_class = num_instances // num_classes
        
        # For each class, extract the uncertainty values.
        for cls in range(num_classes):
            start_idx = cls * num_per_class
            end_idx = (cls + 1) * num_per_class
            values = df.loc[start_idx:end_idx - 1, uncertainty_metric].values
            
            # The first 4 classes are considered in-distribution (ID).
            if cls < 4:
                uncertainties_ID.extend(values)
            # For OOD, include only the classes specified by selected_ood_classes.
            if cls in selected_ood_classes:
                uncertainties_OOD.extend(values)

    # Convert lists to NumPy arrays for further processing.
    uncertainties_ID = np.array(uncertainties_ID)
    uncertainties_OOD = np.array(uncertainties_OOD)
    
    # Begin plotting the histograms.
    plt.figure(figsize=(10, 6))
    bins = 30  # Define number of bins for the histograms
    
    # Plot histogram for in-distribution uncertainties (ID)
    plt.hist(uncertainties_ID, bins=bins, density=True, histtype='step', linewidth=2,
             label='In-Distribution')
    
    # Check if OOD uncertainties exist before plotting.
    if uncertainties_OOD.size > 0:
        # Plot histogram for out-of-distribution uncertainties (OOD)
        plt.hist(uncertainties_OOD, bins=bins, density=True, histtype='step', linewidth=2,
             label='Out-of-Distribution')
        # # Plot vertical lines for the 5th and 95th percentiles for OOD
        # p5_OOD, p95_OOD = np.percentile(uncertainties_OOD, [5, 95])
        # plt.axvline(p5_OOD, color='orange', linestyle='--', label='OOD 5th Percentile')
        # plt.axvline(p95_OOD, color='orange', linestyle=':', label='OOD 95th Percentile')
    
    # # Compute the 5th and 95th percentiles for each group.
    # p5_ID, p95_ID = np.percentile(uncertainties_ID, [5, 95])
    # # Plot vertical lines for the 5th and 95th percentiles for ID
    # plt.axvline(p5_ID, color='blue', linestyle='--', label='ID 5th Percentile')
    # plt.axvline(p95_ID, color='blue', linestyle=':', label='ID 95th Percentile')
    
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title('Uncertainty Histograms')
    plt.legend()
    plt.tight_layout()
    
    # Save and display the plot.
    selected_str = "".join([hparams.class_names[i] for i in selected_ood_classes])
    plt.savefig(os.path.join(hparams.model_dir, f"bnn_ID_OOD{selected_str}_uncertainty_histograms.png"))
    plt.show()
    plt.close()


def plot_IDODD_KDE(hparams, num_trials=10, uncertainty_metric='epistemic_depeweg', selected_ood_classes=None):
    """
    Plots smooth Kernel Density Estimation (KDE) distributions for in-distribution (ID) and 
    out-of-distribution (OOD) uncertainty values. The areas under the curves are filled 
    (green for ID, red for OOD) to improve readability.
    
    Parameters:
        hparams: Hyperparameter object that should include:
            - model_dir: Base directory for model results.
            - num_classes: Total number of classes.
            - class_names: List of class names.
            - mc_uncertainty_directory: Directory where uncertainty CSVs are stored.
        num_trials: Number of trials to aggregate over.
        uncertainty_metric: Column name in CSV to use for uncertainty (default 'epistemic_depeweg').
        selected_ood_classes: List of class indices to include in OOD plot (default: all OOD classes).
    
    CSV file format:
        - "saved_bnn_testset_uncertainty_metrics_trial_{trial}.csv"
        - The first 4 classes (indices 0-3) are considered ID.
    """
    
    num_classes = hparams.num_classes
    uncertainty_directory = hparams.mc_uncertainty_directory  

    # Default: If no specific OOD classes provided, include all OOD classes (index ≥ 4)
    if selected_ood_classes is None:
        selected_ood_classes = list(range(4, num_classes))
    
    # Lists to store uncertainty values for ID and selected OOD classes
    uncertainties_ID = []
    uncertainties_OOD = []

    # Loop over trials and collect uncertainty values
    for trial in range(num_trials):
        file_path = os.path.join(
            hparams.model_dir,
            uncertainty_directory,
            f"saved_bnn_testset_uncertainty_metrics_trial_{trial}.csv"
        )
        df = pd.read_csv(file_path)
        
        # Determine number of instances per class (assumes equal distribution)
        num_instances = len(df)
        num_per_class = num_instances // num_classes
        
        # Extract uncertainty values
        for cls in range(num_classes):
            start_idx = cls * num_per_class
            end_idx = (cls + 1) * num_per_class
            values = df.loc[start_idx:end_idx - 1, uncertainty_metric].values
            
            if cls < 4:
                uncertainties_ID.extend(values)
            if cls in selected_ood_classes:
                uncertainties_OOD.extend(values)

    # Convert lists to NumPy arrays
    uncertainties_ID = np.array(uncertainties_ID)
    uncertainties_OOD = np.array(uncertainties_OOD)

    # Define KDE for smooth curves
    kde_ID = gaussian_kde(uncertainties_ID)
    x_ID = np.linspace(min(uncertainties_ID), max(uncertainties_ID), 200)
    y_ID = kde_ID(x_ID)

    # Plot KDE curves with filled areas
    plt.figure(figsize=(10, 6))

    # Plot ID distribution (Green)
    plt.fill_between(x_ID, y_ID, color='green', alpha=0.5, label='In-Distribution (ID)')
    plt.plot(x_ID, y_ID, color='green', linewidth=2)

    # Plot OOD distribution (Red) if available
    if uncertainties_OOD.size > 0:
        kde_OOD = gaussian_kde(uncertainties_OOD)
        x_OOD = np.linspace(min(uncertainties_OOD), max(uncertainties_OOD), 200)
        y_OOD = kde_OOD(x_OOD)
        plt.fill_between(x_OOD, y_OOD, color='red', alpha=0.5, label='Out-of-Distribution (OOD)')
        plt.plot(x_OOD, y_OOD, color='red', linewidth=2)

        # p5_OOD, p95_OOD = np.percentile(uncertainties_OOD, [5, 95])
        # plt.axvline(p5_OOD, color='darkred', linestyle='--', label='OOD 5th Percentile')
        # plt.axvline(p95_OOD, color='darkred', linestyle=':', label='OOD 95th Percentile')

    # # Compute 5th and 95th percentiles for ID
    # p5_ID, p95_ID = np.percentile(uncertainties_ID, [5, 95])
    # # Add percentile lines for ID
    # plt.axvline(p5_ID, color='darkgreen', linestyle='--', label='ID 5th Percentile')
    # plt.axvline(p95_ID, color='darkgreen', linestyle=':', label='ID 95th Percentile')

    # Formatting
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title('Smoothed Uncertainty Distributions')
    plt.legend()
    plt.tight_layout()

    # Filename based on OOD class names
    selected_str = "".join([hparams.class_names[i] for i in selected_ood_classes])
    plt.savefig(os.path.join(hparams.model_dir, f"bnn_ID_OOD{selected_str}_uncertainty_KDE.png"))
    plt.show()
    plt.close()


def plot_bnn_bar_plot(hparams, num_trials=10, uncertainty_metric="depeweg"):
    """
    Plots BNN uncertainty statistics for each class.
    
    For uncertainty_metric="depeweg" (default), the function:
      - Extracts "avg_epistemic_depeweg" and "avg_aleatoric_depeweg" from each trial.
      - Computes the mean and 5th/95th percentile-based error bars for each class.
      - Plots side-by-side bar plots for epistemic and aleatoric uncertainties.
    
    For uncertainty_metric="entropy":
      - Extracts "avg_entropy_fischer" from each trial.
      - Computes the mean and 5th/95th percentile-based error bars for each class.
      - Normalizes these values by dividing by the largest class mean (so that the maximum becomes 1).
      - Plots a single bar plot with error bars.
    
    In both cases, the function saves:
      - A CSV file named "saved_bnn_testset_class_average_uncertainty_statistics.csv" 
        containing the calculated means and error bars.
      - The figure as "bnn_bar_plot_{uncertainty_metric}.png" in the model directory.
    
    Parameters:
        hparams: Hyperparameter object that should include:
            - model_dir: Base directory for model results.
            - num_classes: Number of classes.
            - class_names: List of class names.
            - mc_uncertainty_directory: Directory name where uncertainty CSVs are stored.
        uncertainty_metric: "depeweg" (default) or "entropy".
        num_trials: Number of Monte Carlo trials to aggregate.
    """
    
    num_classes = hparams.num_classes
    class_names = hparams.class_names
    uncertainty_directory = hparams.mc_uncertainty_directory
    
    # Dictionaries to store uncertainty values for each class over trials.
    if uncertainty_metric == "depeweg":
        epistemic_values = {cls: [] for cls in range(num_classes)}
        aleatoric_values = {cls: [] for cls in range(num_classes)}
    elif uncertainty_metric == "entropy":
        entropy_values = {cls: [] for cls in range(num_classes)}
    else:
        raise ValueError("Unsupported uncertainty_metric. Choose 'depeweg' or 'entropy'.")
    
    # Loop over trials to read CSV files.
    for trial in range(num_trials):
        file_path = os.path.join(
            hparams.model_dir,
            uncertainty_directory,
            f"saved_bnn_testset_class_average_uncertainty_trial_{trial}.csv"
        )
        df = pd.read_csv(file_path)
        # Assumes each CSV file has one row per class in the same order as hparams.class_names.
        for cls in range(num_classes):
            if uncertainty_metric == "depeweg":
                epistemic_val = df.iloc[cls]["avg_epistemic_depeweg"]
                aleatoric_val = df.iloc[cls]["avg_aleatoric_depeweg"]
                epistemic_values[cls].append(epistemic_val)
                aleatoric_values[cls].append(aleatoric_val)
            elif uncertainty_metric == "entropy":
                entropy_val = df.iloc[cls]["avg_entropy_fischer"]
                entropy_values[cls].append(entropy_val)
    
    # Initialize arrays to store statistics.
    if uncertainty_metric == "depeweg":
        mean_epistemic = np.zeros(num_classes)
        lower_epistemic = np.zeros(num_classes)
        upper_epistemic = np.zeros(num_classes)
        
        mean_aleatoric = np.zeros(num_classes)
        lower_aleatoric = np.zeros(num_classes)
        upper_aleatoric = np.zeros(num_classes)
        
        # Calculate statistics for each class.
        for cls in range(num_classes):
            e_vals = np.array(epistemic_values[cls])
            a_vals = np.array(aleatoric_values[cls])
            
            mean_epistemic[cls] = np.mean(e_vals)
            lower_epistemic[cls] = np.percentile(e_vals, 5)
            upper_epistemic[cls] = np.percentile(e_vals, 95)
            
            mean_aleatoric[cls] = np.mean(a_vals)
            lower_aleatoric[cls] = np.percentile(a_vals, 5)
            upper_aleatoric[cls] = np.percentile(a_vals, 95)
        
        # Compute error bars.
        epistemic_err_lower = mean_epistemic - lower_epistemic
        epistemic_err_upper = upper_epistemic - mean_epistemic
        aleatoric_err_lower = mean_aleatoric - lower_aleatoric
        aleatoric_err_upper = upper_aleatoric - mean_aleatoric
        
        # Plot side-by-side bar plot.
        x = np.arange(num_classes)
        bar_width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Epistemic uncertainty bars.
        ax.bar(x - bar_width/2, mean_epistemic, bar_width, 
               yerr=[epistemic_err_lower, epistemic_err_upper],
               capsize=5, label='Epistemic')
        # Aleatoric uncertainty bars.
        ax.bar(x + bar_width/2, mean_aleatoric, bar_width,
               yerr=[aleatoric_err_lower, aleatoric_err_upper],
               capsize=5, label='Aleatoric')
        
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylabel('Average Uncertainty')
        ax.set_title('BNN Average Uncertainty vs. Tactic')
        ax.legend()
        
        # Save computed statistics in a DataFrame.
        stats_df = pd.DataFrame({
            "class": class_names,
            "mean_epistemic": mean_epistemic,
            "err_lower_epistemic": epistemic_err_lower,
            "err_upper_epistemic": epistemic_err_upper,
            "mean_aleatoric": mean_aleatoric,
            "err_lower_aleatoric": aleatoric_err_lower,
            "err_upper_aleatoric": aleatoric_err_upper
        })
    elif uncertainty_metric == "entropy":
        mean_entropy = np.zeros(num_classes)
        lower_entropy = np.zeros(num_classes)
        upper_entropy = np.zeros(num_classes)
        
        for cls in range(num_classes):
            vals = np.array(entropy_values[cls])
            mean_entropy[cls] = np.mean(vals)
            lower_entropy[cls] = np.percentile(vals, 5)
            upper_entropy[cls] = np.percentile(vals, 95)
        
        # Compute error bars.
        entropy_err_lower = mean_entropy - lower_entropy
        entropy_err_upper = upper_entropy - mean_entropy
        
        # Normalize values so that the maximum mean becomes 1.
        max_mean = np.max(mean_entropy)
        norm_mean_entropy = mean_entropy / max_mean
        norm_err_lower = entropy_err_lower / max_mean
        norm_err_upper = entropy_err_upper / max_mean
        
        # Plot single bar plot.
        x = np.arange(num_classes)
        bar_width = 0.6  # full width for single bars
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, norm_mean_entropy, bar_width,
               yerr=[norm_err_lower, norm_err_upper],
               capsize=5, label='Normalized Entropy')
        
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylabel('Normalized Average Entropy')
        ax.set_title('BNN Average Entropy vs. Tactic')
        ax.legend()
        
        # Save computed statistics in a DataFrame.
        stats_df = pd.DataFrame({
            "class": class_names,
            "max_normalize_mean_entropy": norm_mean_entropy,
            "max_normalize_err_lower_entropy": norm_err_lower,
            "max_normalize_err_upper_entropy": norm_err_upper
        })
    
    # Save the plot with updated file name.
    plt.tight_layout()
    plt.savefig(os.path.join(hparams.model_dir, f"bnn_bar_plot_{uncertainty_metric}.png"))
    plt.show()
    plt.close()
    
    # Save the computed statistics to a CSV file in the uncertainty directory.
    csv_path = os.path.join(
        hparams.model_dir,
        uncertainty_directory,
        f"saved_bnn_testset_class_average_uncertainty_{uncertainty_metric}_statistics.csv"
        )
    stats_df.to_csv(csv_path, index=False)


def plot_dnn_bar_plot(hparams):
    """
    Plots a bar chart of DNN uncertainties for each class using the average entropy values
    normalized by the maximum value (all class averages / max average value).
    
    Parameters:
        hparams: Hyperparameter object with the following attributes:
            - model_dir: Base directory for model results.
            - num_classes: Number of classes.
            - class_names: List of class names.
    """
    
    num_classes = hparams.num_classes
    class_names = hparams.class_names
    
    # Construct the path to the CSV file containing the DNN uncertainties.
    file_path = os.path.join(hparams.model_dir, "saved_dnn_testset_class_average_uncertainty.csv")
    # Read the CSV file.
    df = pd.read_csv(file_path)
    
    # Extract the average entropy values for each class.
    mean_entropy = np.zeros(num_classes)
    for cls in range(num_classes):
        mean_entropy[cls] = df.iloc[cls]["avg_entropy_fischer"]
    
    # Normalize the entropy values so that the maximum mean becomes 1.
    max_mean = np.max(mean_entropy)
    norm_mean_entropy = mean_entropy / max_mean
    
    # Create a bar plot.
    x = np.arange(num_classes)
    bar_width = 0.6  # full width for single bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, norm_mean_entropy, bar_width, label='Normalized Entropy')
    
    # Labeling the plot.
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel('Normalized Average Entropy')
    ax.set_title('DNN Average Entropy vs. Tactic')
    ax.legend()
    
    plt.tight_layout()
    
    # Save the plot with the updated file name.
    plot_path = os.path.join(hparams.model_dir, "dnn_bar_plot_entropy.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    # Save the computed statistics to a CSV file.
    stats_df = pd.DataFrame({
        "class": class_names,
        "max_normalize_mean_entropy": norm_mean_entropy
    })
    csv_path = os.path.join(hparams.model_dir, "saved_dnn_testset_class_average_entropy_statistics.csv")
    stats_df.to_csv(csv_path, index=False)


def plot_class_probabilities(hparams, instance_number=0):
    '''
    Selects the true label "instance number" (ie. 0=first, 1=second, etc) for each class and plots class probabilities
    and uncertainty metrics, which are also saved as a .CSV.
    For BNN, tight grouping shows low uncertainty (high confidence), while large spread shows low confidence.
    
    Note: hparams.num_classes is the total number of classes in the test set (can be any number of classes),
          while the model outputs only num_classes_pred (e.g., 4).
    '''
    
    # ==========================================================
    # Load variables required
    # ==========================================================
    num_classes = hparams.num_classes                   # number of inference classes presented to NN during inference
    class_names = hparams.class_names                   # class names presented to NN during inference
    class_names_pred = hparams.class_names_pred         # class names NN trained on
    
    # Based on testset seedstart, will print seed number based on instance_number
    startseed = 7501
    seed = startseed + instance_number

    # Determine the correct file paths
    if hparams.bnn:
        file_suffix = "bnn"
        mc_directory = hparams.mc_sample_directory
        uncertainty_directory = hparams.mc_uncertainty_directory
        trial = 0       # arbitrarily set to first MC trial
        predictions_path = os.path.join(hparams.model_dir, mc_directory, f"saved_bnn_mc_predictions_trial_{trial}.npy")
        uncertainty_path = os.path.join(hparams.model_dir, uncertainty_directory, f"saved_bnn_testset_uncertainty_metrics_trial_{trial}.csv")
        class_avg_path = os.path.join(hparams.model_dir, uncertainty_directory, f"saved_{file_suffix}_testset_class_average_uncertainty_trial_{trial}.csv")
    else:
        file_suffix = "dnn"
        predictions_path = os.path.join(hparams.model_dir, f"saved_{file_suffix}_predictions.npy")
        uncertainty_path = os.path.join(hparams.model_dir, f"saved_{file_suffix}_testset_uncertainty_metrics.csv")
        class_avg_path = os.path.join(hparams.model_dir, f"saved_{file_suffix}_testset_class_average_uncertainty.csv")
    
    if not os.path.exists(predictions_path) or not os.path.exists(uncertainty_path):
        print(f"[WARNING] Missing files for class probability plot: {predictions_path}, {uncertainty_path}")
        return

    # Load predictions and uncertainty metrics
    predictions = np.load(predictions_path)             # Shape: BNN=[MC_samples, num_test_samples, num_classes]; DNN=[num_test_samples, num_classes]
    uncertainty_df = pd.read_csv(uncertainty_path)      # Shape: [num_test_samples, num_metrics]
    class_avgs = pd.read_csv(class_avg_path)            # Load class averages computed in show_uncertainty
    true_labels = uncertainty_df['true_label'].values   # Shape: [num_test_samples]
    
    num_classes_pred = predictions.shape[-1]            # number of class predictions (based on NN output size)
    
    # Ensure num_test_samples true labels and predictions align
    assert len(true_labels) == predictions.shape[-2], "Mismatch between number of true labels and test samples!"

    ## NOT USED: but could be used to included predicted class in plot title
    # Compute single predicted label for each instance of the test set.
    # For BNN, we average over the MC samples and then take argmax.
    if hparams.bnn:
        y_pred = np.argmax(predictions.mean(axis=0), axis=-1)
    else:
        y_pred = np.argmax(predictions, axis=-1)

    # Prepare a figure with subplots for each class
    # Create a grid of subplots that is as square as possible
    n_cols = math.ceil(math.sqrt(num_classes))
    n_rows = math.ceil(num_classes / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 12))
    # Flatten axes array for easy indexing (if there's only one subplot, wrap it in a list)
    axes = axes.flatten() if num_classes > 1 else [axes]
    # Choose a constant row height (in axis coordinates) for the tables:
    row_height = 0.05  # Adjust as needed for table display

    # Loop over each class and plot the corresponding instance predictions and uncertainties
    for cls in range(num_classes):
        indices = np.where(true_labels == cls)[0]                       # Finds all instances where the true label matches the current class; [0] ensures 1D
        if len(indices) == 0:
            print(f"No test instances found for class {class_names[cls]}")
            continue
        if len(indices) <= instance_number:
            print(f"[WARNING] Not enough test instances for class {class_names[cls]}. Skipping.")
            continue
        instance_idx = indices[instance_number]                         # Pick "instance_number" (1st, 2nd, 3rd, etc) instance with this true label
        instance_predictions = predictions[:, instance_idx, :] if hparams.bnn else predictions[instance_idx, :] # (BNN) Extract Monte Carlo predictions for this instance; Shape: [MC_samples, num_classes]; (DNN) extract class predictions

        # Extract uncertainties for this instance
        instance_metrics = uncertainty_df.iloc[instance_idx]
        entropy = instance_metrics['entropy_fischer']
        aleatoric = instance_metrics['aleatoric_depeweg'] if hparams.bnn else instance_metrics['aleatoric_kwon']
        epistemic = instance_metrics['epistemic_depeweg'] if hparams.bnn else None

        # Retrieve precomputed class averages for this class
        row = class_avgs[class_avgs['class'] == class_names[cls]]
        avg_entropy = row['avg_entropy_fischer'].values[0]
        avg_accuracy = row['avg_accuracy'].values[0]
        if hparams.bnn:
            avg_aleatoric_depeweg = row['avg_aleatoric_depeweg'].values[0]
            avg_epistemic_depeweg = row['avg_epistemic_depeweg'].values[0]
        else:
            avg_aleatoric_kwon = row['avg_aleatoric_kwon'].values[0]

        # Get the predicted probabilities
        pred_probs = instance_predictions.mean(axis=0) if hparams.bnn else instance_predictions  # Mean over MC samples for BNN

        # Plot class probabilities (only for classes model trained on; num_classes_pred)
        ax = axes[cls]
        for class_idx in range(num_classes_pred):
            probabilities = instance_predictions[:, class_idx] if hparams.bnn else [instance_predictions[class_idx]]
            ax.scatter(
                [class_idx] * len(probabilities),   # Class index on x-axis (will match classes model trained on)
                probabilities,                      # Probabilities on y-axis
                alpha=0.7,                          # Marker transparency
                s=30                                # Marker size
                # label=f"Class {class_idx}" if class_idx == 0 else None
            )
        ax.set_ylim(-0.05, 1.05)   # fix y axis 0-1

        # Build a table to display instance uncertainties and class averages.
        if hparams.bnn:
            table_data = [
                ["Metric", "Value"],
                ["Entropy", f"{entropy:.2f}"],
                ["Aleatoric", f"{aleatoric:.2f}"],
                ["Epistemic", f"{epistemic:.2f}"],
                ["Cls Avg...", " "],
                ["...Ent", f"{avg_entropy:.2f}"],
                ["...Ale", f"{avg_aleatoric_depeweg:.2f}"],
                ["...Epi", f"{avg_epistemic_depeweg:.2f}"],
                ["...Acc", f"{avg_accuracy:.2f}"]
                ]
        else:
            table_data = [
            ["Metric", "Value"],
            ["Entropy", f"{entropy:.2f}"],
            ["Aleatoric", f"{aleatoric:.2f}"],
            ["Cls Avg...", " "],
            ["...Ent", f"{avg_entropy:.2f}"],
            ["...Ale", f"{avg_aleatoric_kwon:.2f}"],
            ["...Acc", f"{avg_accuracy:.2f}"]
            ]
        num_rows_uncertainty = len(table_data)
        height_uncertainty = num_rows_uncertainty * row_height
        table = ax.table(
            cellText=table_data,
            colLabels=None,
            cellLoc="center",
            colWidths=[0.3, 0.2],
            loc="upper right",
            bbox=[0.68, 0.5, 0.25, height_uncertainty] # [x0, y0, width, height]
        )

        # Disable autosizing and set the font size
        table.auto_set_font_size(False)
        table.set_fontsize(7)  # Increase the font size (adjust as needed)

        # Highlight specific rows with target text in the first column.
        for r, row_data in enumerate(table_data):
            if row_data[0] == "...Epi" or row_data[0] == "Epistemic":
                # For the matching row, highlight all cells in that row.
                for c in range(len(row_data)):
                    table.get_celld()[(r, c)].set_facecolor('lightyellow')

        # Add a second table below the first one for predicted class probabilities
        pred_table = [[class_names_pred[i], f"{pred_probs[i]:.2f}"] for i in range(num_classes_pred)]
        num_rows_pred = len(pred_table)
        height_pred = (1 + num_rows_pred) * row_height # add 1 for column labels row
        table = ax.table(
            cellText=pred_table,
            colLabels=["Class", "Prob"],
            cellLoc="center",
            colWidths=[0.3, 0.2],
            loc="upper right",
            bbox=[0.68, 0.4-height_pred, 0.25, height_pred]  # Slightly below the first table
        )

        # Disable autosizing and set the font size
        table.auto_set_font_size(False)
        table.set_fontsize(7)  # Increase the font size (adjust as needed)
        
        # Format the subplot
        ax.set_xticks(range(num_classes_pred))
        ax.set_xticklabels(class_names_pred[:num_classes_pred])
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        # ax.set_title(f"True Label: {class_names[cls]} (Instance {instance_idx})")
        ax.set_title(f"True Label: {class_names[cls]} (Seed {seed})")
        ax.grid(True)

    # Turn off any unused subplots (if num_classes < n_rows*n_cols)
    for j in range(num_classes, len(axes)):
        axes[j].axis('off')    

    # Adjust layout and save the figure
    plt.tight_layout()
    # plt.savefig(os.path.join(hparams.model_dir, f"BNN_test_instance_{instance_number}_MC_predictions_by_true_class.png"))
    plt.savefig(os.path.join(hparams.model_dir, f"{file_suffix}_prediction_test_instance_{instance_number}_seed_{seed}.png"))
    plt.show()
    plt.close()


def plot_vs_data_retained(hparams, num_trials=10, uncertainty_metric='entropy_fischer'):
    """
    Plots multiclass accuracy and class data retained vs. percentage of test instances retained, 
    based on the specified uncertainty metric used in sort_and_calc function.
    
    For DNN:
        - Plots the single accuracy and class retention curves.
    For BNN:
        - Computes the mean and the 5th/95th percentiles (confidence intervals) across trials,
          and plots the mean curves along with the confidence interval using plt.fill_between.
          
    Also, saves a CSV file containing the mean and confidence interval (CI) values for:
        - Accuracy: mean, lower (5th percentile), and upper (95th percentile)
        - For each class, the retained percentage: mean, lower, and upper values.
       
    Parameters:
        hparams: Object containing the model directory path and other hyperparameters.
        num_trials: number of MC trials.
    """

    # ==========================================================
    # Load variables required
    # ==========================================================
    num_classes = hparams.num_classes                   # number of inference classes presented to NN during inference
    class_names = hparams.class_names                   # class names presented to NN during inference

    # ==========================================================
    # Load Data and Calculate Statistics
    # ==========================================================
    if hparams.bnn:
        file_suffix = "bnn"
        uncertainty_directory = hparams.mc_uncertainty_directory

        # Lists to store accuracy arrays for each trial and dictionary for class retention.
        all_trials_accuracy = []  
        all_trials_class_retention = {cls: [] for cls in range(num_classes)}
        
        for trial in range(num_trials):
            sorted_uncertainty_path = os.path.join(
                hparams.model_dir,
                uncertainty_directory,
                f"saved_{file_suffix}_testset_uncertainty_metrics_{uncertainty_metric}_sorted_trial_{trial}.csv"
            )
            sorted_uncertainty_df = pd.read_csv(sorted_uncertainty_path)
            
            trial_accuracy = np.array(sorted_uncertainty_df["accuracy_percent"].values)
            all_trials_accuracy.append(trial_accuracy)
            
            for cls in range(num_classes):
                col_name = f"{class_names[cls]}_retained_percent"
                trial_retention = np.array(sorted_uncertainty_df[col_name].values)
                all_trials_class_retention[cls].append(trial_retention)
        
        # Convert lists to numpy arrays for vectorized operations.
        all_trials_accuracy = np.array(all_trials_accuracy)  # shape: (num_trials, num_points)
        
        # Compute statistics for accuracy:
        mean_accuracy = np.mean(all_trials_accuracy, axis=0)
        lower_accuracy = np.percentile(all_trials_accuracy, 5, axis=0)
        upper_accuracy = np.percentile(all_trials_accuracy, 95, axis=0)
        
        # For class retention, compute mean and CI for each class
        mean_class_retention = {}
        lower_class_retention = {}
        upper_class_retention = {}
        for cls in range(num_classes):
            data = np.array(all_trials_class_retention[cls])  # shape: (num_trials, num_points)
            mean_class_retention[cls] = np.mean(data, axis=0)
            lower_class_retention[cls] = np.percentile(data, 5, axis=0)
            upper_class_retention[cls] = np.percentile(data, 95, axis=0)
    
    else:
        file_suffix = "dnn"
        sorted_uncertainty_path = os.path.join(
            hparams.model_dir,
            f"saved_{file_suffix}_testset_uncertainty_metrics_{uncertainty_metric}_sorted.csv"
        )
        sorted_uncertainty_df = pd.read_csv(sorted_uncertainty_path)
        mean_accuracy = np.array(sorted_uncertainty_df["accuracy_percent"].values)
        # For DNN, no confidence intervals are available so we assign the same values.
        lower_accuracy = mean_accuracy
        upper_accuracy = mean_accuracy

        mean_class_retention = {}
        lower_class_retention = {}
        upper_class_retention = {}
        for cls in range(num_classes):
            col_name = f"{class_names[cls]}_retained_percent"
            mean_class_retention[cls] = np.array(sorted_uncertainty_df[col_name].values)
            lower_class_retention[cls] = mean_class_retention[cls]
            upper_class_retention[cls] = mean_class_retention[cls]
    
    # Retrieve x-axis values (common to all sorted df of same length)
    data_retained_percent = np.array(sorted_uncertainty_df["data_retained_percent"].values)

    # ==========================================================
    # Plotting
    # ==========================================================
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Primary Y-Axis: Accuracy vs. Data Retained
    color = 'blue'
    ax1.plot(data_retained_percent, mean_accuracy, linestyle='-', linewidth=2,
             label=f"{uncertainty_metric}", color=color)
    
    # For BNN, add confidence interval shading for accuracy using plt.fill_between:
    if hparams.bnn:
        ax1.fill_between(data_retained_percent, lower_accuracy, upper_accuracy, color=color, alpha=0.3,
                         label='5/95 CI (Accuracy)')
    
    ax1.set_xlabel('% of Test Instances Retained')
    ax1.set_ylabel('Accuracy [%]', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title(f'Accuracy vs. Data Retained (sorted by {uncertainty_metric})')
    
    # Secondary Y-Axis: Class Retention
    ax2 = ax1.twinx()
    ax2.set_ylabel('% of Class Instances Retained')
    
    # Define colors for classes
    base_colors = ['red', 'orange', 'purple', 'green'] # in-distribution class colors
    num_colors = num_classes
    num_additional = num_colors - len(base_colors)
    cmap = cm.get_cmap('viridis', num_additional)
    additional_colors = [mcolors.to_hex(cmap(i)) for i in range(num_additional)]
    class_colors = base_colors + additional_colors
    
    for cls, cls_color in zip(range(num_classes), class_colors):
        # Plot mean retention curve for each class
        ax2.plot(data_retained_percent, mean_class_retention[cls], linestyle='--', linewidth=1.5,
                 label=class_names[cls], color=cls_color)
        # For BNN, add fill for confidence interval
        if hparams.bnn:
            ax2.fill_between(data_retained_percent,
                             lower_class_retention[cls],
                             upper_class_retention[cls],
                             color=cls_color, alpha=0.3)
    
    ax2.tick_params(axis='y')
    
    # Invert X-axis for both axes (from 100 to 50)
    ax1.set_xlim(100, 50)
    ax2.set_xlim(100, 50)
    
    # Add legends and grid settings
    ax2.legend(loc='lower center', frameon=True, facecolor="white", edgecolor="black")
    ax1.yaxis.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color=color)
    ax2.grid(visible=True, which='major', linestyle='--', linewidth=0.5)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax2.grid(visible=True, which='minor', linestyle=':', linewidth=0.25)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax1.xaxis.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='black')
    ax1.xaxis.grid(visible=True, which='minor', linestyle=':', linewidth=0.25)
    
    fig.tight_layout()
    ax1.set_ylim(min(min(mean_accuracy), 90), 100)  # 90 is good min to compare in-distribution curves; min mean for OOD
    ax2.set_ylim(0, 100)
    
    # Save the plot
    plt.savefig(
        os.path.join(
            hparams.model_dir,
            f"{file_suffix}_accuracy_vs_data_retained_{uncertainty_metric}.png"
        ))
    plt.show()
    plt.close()

    # ==========================================================
    # Saving Calculated Statistics as CSV
    # ==========================================================
    # Create a DataFrame with the x-axis and accuracy statistics.
    stats_df = pd.DataFrame({
        "data_retained_percent": data_retained_percent,
        "accuracy_mean": mean_accuracy,
        "accuracy_lower": lower_accuracy,
        "accuracy_upper": upper_accuracy
    })
    
    # For each class, add mean, lower and upper retention statistics.
    for cls in range(num_classes):
        stats_df[f"{class_names[cls]}_retained_mean"] = mean_class_retention[cls]
        stats_df[f"{class_names[cls]}_retained_lower"] = lower_class_retention[cls]
        stats_df[f"{class_names[cls]}_retained_upper"] = upper_class_retention[cls]
    
    # Determine the directory for saving based on model type.
    if hparams.bnn:
        csv_save_dir = os.path.join(hparams.model_dir, uncertainty_directory)
    else:
        csv_save_dir = hparams.model_dir

    csv_path = os.path.join(csv_save_dir, 
               f"saved_{file_suffix}_testset_uncertainty_metrics_{uncertainty_metric}_sorted_statistics.csv")
    stats_df.to_csv(csv_path, index=False)


def plot_vs_data_retained_all_metrics_aggregated(hparams, uncertainty_metrics):
    """
    Loads the statistics CSV files for a list of uncertainty metrics, aggregates the accuracy and class
    retention curves across metrics, and plots a single figure with:
      - Primary Y-Axis (ax1): Aggregated accuracy (mean with 5/95 CI) vs. data retained.
      - Secondary Y-Axis (ax2): For each class, aggregated retention curves (mean with 5/95 CI) vs. data retained.
    Additionally, saves the aggregated statistics (means and 5/95 CI) for accuracy and class retention as a CSV file.

    Parameters:
        hparams: Object with attributes:
            - model_dir: Base directory.
            - mc_uncertainty_directory: For BNN, the subdirectory where CSV files are stored.
            - num_classes: Number of classes.
            - class_names: List of class names.
            - bnn: Boolean flag indicating if BNN is used.
        uncertainty_metrics: List of uncertainty metric strings, e.g.
            ['entropy_fischer', 'epistemic_kwon', 'aleatoric_kwon', 'epistemic_depeweg', 'aleatoric_depeweg']
    """
    # Determine the base directory and file suffix for the CSV files.
    if hparams.bnn:
        base_dir = os.path.join(hparams.model_dir, hparams.mc_uncertainty_directory)
        file_suffix = "bnn"
    else:
        base_dir = hparams.model_dir
        file_suffix = "dnn"

    # Load CSV for the first uncertainty metric to get the common x-axis.
    sample_csv = os.path.join(base_dir, f"saved_{file_suffix}_testset_uncertainty_metrics_{uncertainty_metrics[0]}_sorted_statistics.csv")
    sample_df = pd.read_csv(sample_csv)
    x = sample_df["data_retained_percent"].values

    num_classes = hparams.num_classes
    class_names = hparams.class_names

    # Initialize lists to store the curves from each uncertainty metric.
    accuracy_means_list = []
    
    # For class retention, create dictionaries keyed by class index.
    retention_means = {cls: [] for cls in range(num_classes)}
    retention_lowers = {cls: [] for cls in range(num_classes)}
    retention_uppers = {cls: [] for cls in range(num_classes)}

    # Loop over each uncertainty metric and load the corresponding CSV.
    for metric in uncertainty_metrics:
        csv_path = os.path.join(base_dir, f"saved_{file_suffix}_testset_uncertainty_metrics_{metric}_sorted_statistics.csv")
        df = pd.read_csv(csv_path)
        
        # Collect accuracy curves.
        accuracy_means_list.append(df["accuracy_mean"].values)
        
        # For each class, collect the retention curves.
        for cls in range(num_classes):
            col_mean = f"{class_names[cls]}_retained_mean"
            col_lower = f"{class_names[cls]}_retained_lower"
            col_upper = f"{class_names[cls]}_retained_upper"
            retention_means[cls].append(df[col_mean].values)
            retention_lowers[cls].append(df[col_lower].values)
            retention_uppers[cls].append(df[col_upper].values)
    
    # Convert lists to numpy arrays for vectorized computation.
    # Each array will have shape (num_metrics, num_points)
    accuracy_means_array = np.array(accuracy_means_list)
    
    # Compute aggregated (across uncertainty metrics) accuracy curves.
    agg_accuracy_mean = np.mean(accuracy_means_array, axis=0)
    agg_accuracy_lower = np.percentile(accuracy_means_array, 5, axis=0)
    agg_accuracy_upper = np.percentile(accuracy_means_array, 95, axis=0)
    
    # For class retention, aggregate for each class.
    agg_retention_mean = {}
    agg_retention_lower = {}
    agg_retention_upper = {}
    for cls in range(num_classes):
        arr_means = np.array(retention_means[cls])
        agg_retention_mean[cls] = np.mean(arr_means, axis=0)
        agg_retention_lower[cls] = np.percentile(arr_means, 5, axis=0)
        agg_retention_upper[cls] = np.percentile(arr_means, 95, axis=0)
    
    # ==========================================================
    # Plotting: One figure with dual y-axes.
    # ==========================================================
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Primary Y-Axis: Plot aggregated accuracy.
    acc_color = 'blue'
    ax1.plot(x, agg_accuracy_mean, linestyle='-', linewidth=2,
             label='Accuracy (aggregated)', color=acc_color)
    ax1.fill_between(x, agg_accuracy_lower, agg_accuracy_upper, color=acc_color, alpha=0.3,
                     label='5/95 CI (Accuracy)')
    ax1.set_xlabel('% of Test Instances Retained')
    ax1.set_ylabel('Accuracy [%]', color=acc_color)
    ax1.tick_params(axis='y', labelcolor=acc_color)
    ax1.set_title('Mean Accuracy & Class Retention vs. Data Retained (Sorted using ALL METRICS)')
    
    # Secondary Y-Axis: Plot aggregated class retention curves.
    ax2 = ax1.twinx()
    ax2.set_ylabel('% of Class Instances Retained')
    
    # Define a set of colors for the classes.
    base_colors = ['red', 'orange', 'purple', 'green']  # in-distribution class colors
    num_colors = num_classes
    num_additional = max(0, num_colors - len(base_colors))
    cmap = cm.get_cmap('viridis', num_additional) if num_additional > 0 else None
    additional_colors = [mcolors.to_hex(cmap(i)) for i in range(num_additional)] if cmap else []
    class_colors = base_colors + additional_colors
    
    for cls, cls_color in zip(range(num_classes), class_colors):
        ax2.plot(x, agg_retention_mean[cls], linestyle='--', linewidth=1.5,
                 label=class_names[cls], color=cls_color)
        ax2.fill_between(x, agg_retention_lower[cls], agg_retention_upper[cls],
                         color=cls_color, alpha=0.3)
    
    ax2.tick_params(axis='y')
    
    # Invert X-axis (from 100 down to 50).
    ax1.set_xlim(100, 50)
    ax2.set_xlim(100, 50)
    
    # Add legends and grid settings.
    ax2.legend(loc='lower center', frameon=True, facecolor="white", edgecolor="black")
    ax1.yaxis.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color=acc_color)
    ax2.grid(visible=True, which='major', linestyle='--', linewidth=0.5)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax2.grid(visible=True, which='minor', linestyle=':', linewidth=0.25)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax1.xaxis.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='black')
    ax1.xaxis.grid(visible=True, which='minor', linestyle=':', linewidth=0.25)
    
    # Set y-axis limits.
    ax1.set_ylim(min(min(agg_accuracy_mean), 90), 100)
    ax2.set_ylim(0, 100)
    
    fig.tight_layout()
    
    # Save the plot in the model directory.
    save_path = os.path.join(hparams.model_dir, f"{file_suffix}_accuracy_vs_data_retained_0_all_metrics_aggregated.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()
    
    # ==========================================================
    # Saving Calculated Statistics as CSV
    # ==========================================================
    # Create a DataFrame with the x-axis and aggregated accuracy statistics.
    stats_df = pd.DataFrame({
        "data_retained_percent": x,
        "accuracy_mean": agg_accuracy_mean,
        "accuracy_lower": agg_accuracy_lower,
        "accuracy_upper": agg_accuracy_upper
    })
    
    # For each class, add aggregated retention statistics.
    for cls in range(num_classes):
        stats_df[f"{class_names[cls]}_retained_mean"] = agg_retention_mean[cls]
        stats_df[f"{class_names[cls]}_retained_lower"] = agg_retention_lower[cls]
        stats_df[f"{class_names[cls]}_retained_upper"] = agg_retention_upper[cls]
    
    # Determine the directory for saving based on model type.
    if hparams.bnn:
        csv_save_dir = os.path.join(hparams.model_dir, hparams.mc_uncertainty_directory)
    else:
        csv_save_dir = hparams.model_dir

    csv_path = os.path.join(csv_save_dir, 
                            f"saved_{file_suffix}_testset_uncertainty_metrics_statistics_across_all_sorted.csv")
    stats_df.to_csv(csv_path, index=False)


def plot_vs_data_retained_all_metrics(hparams, uncertainty_metrics):
    """
    Plots multiclass accuracy vs. percentage of test instances retained for all specified uncertainty metrics.
    Uses pre-sorted and pre-calculated CSV files for both DNN and BNN models.

    Parameters:
        hparams: Object containing at least the following attributes:
            - model_dir: Directory where model and CSV files are stored.
            - mc_uncertainty_directory: (for BNN) Sub-directory with uncertainty CSV files.
            - bnn: Boolean flag indicating if the model is a BNN.
        uncertainty_metrics: List of uncertainty metric names (strings).
    """
    # Initialize the plot
    plt.figure(figsize=(8, 6))
    min_acc=90

    # Loop through each uncertainty metric to load and plot its CSV file
    for uncertainty_metric in uncertainty_metrics:
        if hparams.bnn:
            file_suffix = "bnn"
            trial = 0  # arbitrarily set to first MC trial
            csv_path = os.path.join(
                hparams.model_dir,
                hparams.mc_uncertainty_directory,
                f"saved_{file_suffix}_testset_uncertainty_metrics_{uncertainty_metric}_sorted_trial_{trial}.csv"
            )
        else:
            file_suffix = "dnn"
            csv_path = os.path.join(
                hparams.model_dir,
                f"saved_{file_suffix}_testset_uncertainty_metrics_{uncertainty_metric}_sorted.csv"
            )
        
        # Check if the file exists; if not, warn and skip this metric
        if not os.path.exists(csv_path):
            print(f"[WARNING] Accuracy plot skipped: {csv_path} not found.")
            continue

        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # It is assumed that the CSV already contains the pre-calculated values:
        # "percentage" -> % of test instances retained
        # "accuracy"   -> Corresponding accuracy in %
        if "data_retained_percent" not in df.columns or "accuracy_percent" not in df.columns:
            print(f"[WARNING] Expected columns 'data_retained_percent' and 'accuracy_percent' not found in {csv_path}. Skipping.")
            continue

        # Plot the pre-calculated curve for the current uncertainty metric
        plt.plot(df["data_retained_percent"], df["accuracy_percent"],
                 linestyle='-', linewidth=2, label=uncertainty_metric)
        
        # undate the minimum accuracy plotted (helps set good y-axis limits later)
        min_acc = min(min_acc, min(df["accuracy_percent"]))

    # Finalize the plot
    plt.gca().invert_xaxis()  # Invert x-axis so that a higher retention percentage is on the left
    plt.gca().set_xlim(100, 50)
    plt.title('Accuracy vs. Data Retained (based on uncertainty metric used for sorting)')
    plt.xlabel('% of Test Instances Retained')
    plt.ylabel('Accuracy [%]')
    plt.gca().set_ylim(min(min_acc, 90), 100)   # set y-axis limits to 90 for comparison across models, or min_acc, as required
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    # plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(10))
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(5))
    plt.legend()
    plt.tight_layout()

    # Save and show the plot
    save_path = os.path.join(hparams.model_dir, f"{file_suffix}_accuracy_vs_data_retained_0_all_metrics.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_training_AccLoss(hparams):
    # Load the logged metrics from CSV
    metrics_filename = "metrics.csv"
    metrics_path = os.path.join(hparams.model_dir, metrics_filename)
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}.")
        return
    metrics = pd.read_csv(metrics_path)
    # Ensure only relevant rows are considered (avoid duplicates)
    metrics = metrics.groupby("epoch").mean().reset_index()
    # Extract epoch numbers
    epochs = metrics["epoch"].values
    # Identify metrics ending in "_loss_epoch" and "_acc_epoch"
    loss_metrics = [col for col in metrics.columns if "loss" in col]
    acc_metrics = [col for col in metrics.columns if "acc" in col]
    # Print minimum validation loss
    val_loss_metrics = [col for col in loss_metrics if "val" in col]
    for metric in val_loss_metrics:
            min_val_loss = metrics[metric].min()
            print(f"Minimum Validation Loss ({metric}): {min_val_loss}")
    # Plot settings
    plt.figure(figsize=(10, 6))
    colormap_loss = cm.Reds(np.linspace(0.3, 0.8, len(loss_metrics)))
    colormap_acc = cm.Blues(np.linspace(0.3, 0.8, len(acc_metrics)))
    # Plot loss metrics
    for color, metric in zip(colormap_loss, loss_metrics):
        label = metric.replace("epoch", "").replace("_", " ").strip().title()
        plt.plot(epochs, metrics[metric], color=color, label=label)
    # Plot accuracy metrics
    for color, metric in zip(colormap_acc, acc_metrics):
        label = metric.replace("epoch", "").replace("_", " ").strip().title()
        plt.plot(epochs, metrics[metric], color=color, label=label)
    # Plot aesthetics
    plt.title("Training & Validation Metrics vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.ylim([0, 1])  ##### MAY NEED [0,2] FOR HIGH LOSS
    plt.legend(loc="best", fontsize=10)
    plt.savefig(os.path.join(hparams.model_dir, "Training_LossAccuracy_vs_Epoch.png"))
    plt.show()


def print_cm(hparams, y_test, y_pred):
    class_names_pred = ["Greedy", "Greedy+", "Auction", "Auction+"]
    # multiclass
    if hparams.output_type == 'mc':
        if hparams.output_length == 'seq':
            y_test=y_test.reshape((-1,1)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred=y_pred.reshape((-1,1)) #flatten both predictions and labels into one column vector each
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=hparams.class_names)
        disp.plot()
        plt.savefig(hparams.model_dir + "Confusion_matrix_MC.png")
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
            plt.savefig(hparams.model_dir + f"Confusion_matrix_ML_{hparams.attribute_names[label_index]}.png")
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
        plt.savefig(hparams.model_dir + "Confusion_matrix_MC.png")
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
            plt.savefig(hparams.model_dir + f"Confusion_matrix_ML_{hparams.attribute_names[label_index]}.png")
            print(f'\nLabel{label_index} {hparams.attribute_names[label_index]}\n',cm[label_index])
        print('\nHamming Loss:',hamming_loss(y_test[1], y_pred[1]),'\n')
        print("\n",classification_report(y_test[1], y_pred[1], target_names=hparams.attribute_names))



# ==========================================================
# CLASS ACTIVATION MAPS
# ==========================================================

# def print_cam(hparams, model, x_train):
#     ''' This function  prints the "Class Activation Map" along with model inputs to help 
#     visualize input importance in making a prediction.  This is only applicable for models
#     that have a Global Average Pooling layer (ex: Fully Convolutional Network)'''
    
#     # select one training sample (engagement) to analyze
#     sample = x_train[6] # 6 = Auction+
#     # Get the class activation map for that sample
#     last_cov_layer=-5 if hparams.output_type == 'mh' else -3 # multihead v2 has 2 extra layers at end
#     heatmap = get_cam(hparams, model, sample, model.layers[last_cov_layer].name)
#     # Save CAM variable for postprocessing
#     append_to_csv(heatmap, hparams.model_dir + "variables.csv", "\nCAM Data")
#     ## Visualize Class Activation Map
#     # ALL FEATURES: Plot the heatmap values along with the time series data for all features (all agents) in that sample
#     plt.figure(figsize=(10, 8))
#     plt.plot(sample, c='black')
#     plt.plot(heatmap, label='CAM [importance]', c='red', lw=5, linestyle='dashed')
#     plt.xlabel('Time Step')
#     plt.ylabel('Feature Value [normalized]')
#     plt.legend()
#     plt.title(f'Class Activation Map vs. All Input Features')
#     plt.savefig(hparams.model_dir + "CAM_all.png")
#     # ONE AGENT: Plot the heatmap values along with the time series features for one agent in that sample
#     plt.figure(figsize=(10, 8))
#     agent_idx=0
#     for feature in range(hparams.num_features_per):
#         plt.plot(sample[:, agent_idx+feature*hparams.num_agents], label=hparams.feature_names[feature])
#     plt.plot(heatmap, label='CAM [importance]', c='red', lw=5, linestyle='dashed')
#     plt.xlabel('Time Step')
#     plt.ylabel('Feature Value [normalized]')
#     plt.legend()
#     plt.title(f"Class Activation Map vs. One Agent's Input Features")
#     plt.savefig(hparams.model_dir + "CAM_one.png")

# def get_cam(hparams, model, sample, last_conv_layer_name):
#     # This function requires the trained FCN model, input sample, and the name of the last convolutional layer
#     # Get the model of the intermediate layers
#     cam_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output[0]]
#         )
#     # Get the last conv_layer outputs and full model predictions
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = cam_model(np.array([sample]))
#         predicted_class = tf.argmax(predictions[0]) # predicted class index
#         predicted_class_val = predictions[:, predicted_class] # predicted class probability
#     # Get the gradients and pooled gradients
#     grads = tape.gradient(predicted_class_val, conv_outputs) # gradients between predicted class probability WRT CONV outputs maps
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1)) # average ("pool") feature gradients (gives single importance value for each map)
#     # Multiply pooled gradients (importance) with the conv layer output, then average across all feature maps, to get the 2D heatmap
#     # Heatmap highlights areas that most influence models prediction
#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
#     # Normalize heatmap
#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap) # normalize values [0,1]
#     # heatmap = heatmap * np.max(sample) # scale to sample maximum
#     # print(f"Heatmap shape {heatmap.shape} \n heatmap values \n {heatmap} ")
#     # append_to_csv(heatmap, hparams.model_dir + "variables.csv", "\nHeatmap")
#     return heatmap[0]



# ==========================================================
# t-Statistical Neighbor Embedding
# ==========================================================

# def print_tsne(hparams, tsne_input, labels, title, perplexity):
#     tsne = TSNE(n_components=2, perplexity=perplexity).fit_transform(tsne_input) # set perplexity = 50-100
#     scaler = MinMaxScaler() # scale between 0 and 1
#     tsne = scaler.fit_transform(tsne.reshape(-1, tsne.shape[-1])).reshape(tsne.shape) # fit amongst tsne_input, then back to original shape
    
#     tx = tsne[:, 0]
#     ty = tsne[:, 1]
    
#     # Dynamically generate colors up to 10
#     # color_list = ['red', 'blue', 'green', 'brown', 'yellow', 'purple', 'orange', 'pink', 'gray', 'cyan']
#     num_classes = len(hparams.class_names)
#     color_map = plt.cm.get_cmap('tab20', num_classes)    # Use the 'tab20' colormap and select the required number of colors
#     colors = [color_map(i) for i in range(num_classes)]  # Generate distinct colors
#     # colors = color_list[:num_classes]
#     # if num_classes > len(color_list):
#     #     raise ValueError(f"Number of classes ({num_classes}) exceeds the number of available colors ({len(color_list)}).")

#     plt.figure()
#     plt.title('tSNE Dimensionality Reduction: '+title)
    
#     for idx, c in enumerate(colors):
#         indices = [i for i, l in enumerate(labels) if idx == l]
#         current_tx = np.take(tx, indices)
#         current_ty = np.take(ty, indices)
#         plt.scatter(current_tx, current_ty, c=c, label=hparams.class_names[idx], alpha=0.5, marker=".")
    
#     plt.legend(loc='best')
#     plt.savefig(hparams.model_dir + "tSNE_"+title+".png")



# ==========================================================
# INPUT-TO-OUTPUT GRADIENT MAP
# ==========================================================

# def compute_and_print_saliency(hparams, model, dataset):
#     # Initialize running sums for the metrics and sensitivity summaries
#     running_total_sensitivity = 0
#     running_sensitivity = 0
#     running_overall_sensitivity = 0
#     running_sensitivity_summary = None
#     num_batches = 0

#     # Iterate through the dataset batch by batch
#     for inputs, labels in dataset:
#         with tf.GradientTape(persistent=True) as tape:
#             tape.watch(inputs)
#             predictions = model(inputs)

#         # Check if predictions are multi-head outputs
#         if isinstance(predictions, list):
#             grads = [tape.jacobian(pred, inputs) for pred in predictions]
#             mean_grads = [tf.reduce_mean(g, axis=[0, 2]) for g in grads]  # Reduce along batch and second axis
#             sensitivity_summaries = [tf.norm(mean_grad, axis=0) for mean_grad in mean_grads]
#         else:
#             grads = tape.jacobian(predictions, inputs)
#             mean_grads = tf.reduce_mean(grads, axis=[0, 2])  # Reduce along batch and second axis
#             sensitivity_summaries = tf.norm(mean_grads, axis=0)

#         # Calculate metrics for each batch
#         if isinstance(sensitivity_summaries, list):
#             for i, sensitivity_summary in enumerate(sensitivity_summaries):
#                 batch_total_sensitivity = tf.reduce_sum(tf.abs(sensitivity_summary))
#                 batch_sensitivity = tf.reduce_sum(tf.abs(mean_grads[i]))
#                 batch_overall_sensitivity = tf.norm(sensitivity_summary)

#                 # Update running totals for multi-head case
#                 running_total_sensitivity += batch_total_sensitivity
#                 running_sensitivity += batch_sensitivity
#                 running_overall_sensitivity += batch_overall_sensitivity

#                 # Accumulate sensitivity summaries for plotting
#                 if running_sensitivity_summary is None:
#                     running_sensitivity_summary = [tf.identity(sensitivity_summary) for sensitivity_summary in sensitivity_summaries]
#                 else:
#                     running_sensitivity_summary[i] += sensitivity_summary

#         else:
#             batch_total_sensitivity = tf.reduce_sum(tf.abs(sensitivity_summaries))
#             batch_sensitivity = tf.reduce_sum(tf.abs(mean_grads))
#             batch_overall_sensitivity = tf.norm(sensitivity_summaries)

#             # Update running totals for single-head case
#             running_total_sensitivity += batch_total_sensitivity
#             running_sensitivity += batch_sensitivity
#             running_overall_sensitivity += batch_overall_sensitivity

#             # Accumulate sensitivity summaries for plotting
#             if running_sensitivity_summary is None:
#                 running_sensitivity_summary = tf.identity(sensitivity_summaries)
#             else:
#                 running_sensitivity_summary += sensitivity_summaries

#         num_batches += 1
#         del grads, tape  # Manually delete to free memory

#     # Average metrics over all batches
#     avg_total_sensitivity = running_total_sensitivity / num_batches
#     avg_sensitivity = running_sensitivity / num_batches
#     avg_overall_sensitivity = running_overall_sensitivity / num_batches

#     # Print or log the aggregated metrics
#     print(f"Average Sum(abs(norm(mean(saliency)))): {avg_total_sensitivity}")
#     print(f"Average Sum(abs(mean(saliency))): {avg_sensitivity}")
#     print(f"Average Norm(mean(saliency)): {avg_overall_sensitivity}")

#     # Plot sensitivity summary
#     if isinstance(running_sensitivity_summary, list):
#         # Multi-head output: Plot each head separately
#         avg_sensitivity_summaries = [s / num_batches for s in running_sensitivity_summary]
#         for i, avg_sensitivity_summary in enumerate(avg_sensitivity_summaries):
#             sensitivity_array = avg_sensitivity_summary.numpy()
#             time_steps = sensitivity_array.shape[0]
#             num_features = sensitivity_array.shape[1]
#             num_features_per = len(hparams.feature_names)
#             num_agents = num_features // num_features_per

#             plt.figure(figsize=(round(num_features / 2), round(time_steps / 2)))
#             ax = sns.heatmap(sensitivity_array[np.newaxis, :], cmap="viridis", cbar_kws={'label': 'Sensitivity'}, square=True)
#             ax.set_title(f'Model Sensitivity Across Input Features for Head {i}')
#             ax.set_xlabel('Input Feature Group')
#             ax.set_ylabel('Time Step')

#             # Set the y-ticks to go from 1 to number of time steps
#             y_ticks = np.arange(0.5, time_steps, 1)  # Start from 0.5 to position in the middle of each square
#             ax.set_yticks(y_ticks)  # Set y-ticks to the center of each row
#             ax.set_yticklabels(np.arange(1, time_steps + 1))  # Labels from 1 to number of time steps

#             # Set the x-ticks to be the middle of each group of features
#             ticks = np.arange(round(num_agents / 2), num_features, num_agents)
#             labels = hparams.feature_names
#             ax.set_xticks(ticks) # Apply custom ticks and labels
#             ax.set_xticklabels(labels, rotation=45) # Rotate labels for better readability
#             for j in range(num_agents, num_features, num_agents):
#                 ax.axvline(x=j, color='white', linestyle='-', linewidth=3)
#             ax.invert_yaxis()

#             plt.savefig(f"{hparams.model_dir}saliency_{hparams.model_type}{hparams.window}_head_{i}.png")

#     else:
#         # Single-head case: Plot the average sensitivity summary
#         avg_sensitivity_summary = running_sensitivity_summary / num_batches
#         sensitivity_array = avg_sensitivity_summary.numpy()
#         time_steps = sensitivity_array.shape[0]
#         num_features = sensitivity_array.shape[1]
#         num_features_per = len(hparams.feature_names)
#         num_agents = num_features // num_features_per

#         plt.figure(figsize=(round(num_features / 2), round(time_steps / 2)))
#         ax = sns.heatmap(sensitivity_array, cmap="viridis", cbar_kws={'label': 'Sensitivity'}, square=True)
#         ax.set_title('Model Sensitivity Across Input Features')
#         ax.set_xlabel('Input Feature Group')
#         ax.set_ylabel('Time Step')

#         # Set the y-ticks to go from 1 to number of time steps
#         y_ticks = np.arange(0.5, time_steps, 1)  # Start from 0.5 to position in the middle of each square
#         ax.set_yticks(y_ticks)  # Set y-ticks to the center of each row
#         ax.set_yticklabels(np.arange(1, time_steps + 1))  # Labels from 1 to number of time steps

#         # Set the x-ticks to be the middle of each group of 10 features
#         ticks = np.arange(round(num_agents / 2), num_features, num_agents) # start/stop/step
#         labels = hparams.feature_names
#         ax.set_xticks(ticks) # Apply custom ticks and labels
#         ax.set_xticklabels(labels, rotation=45) # Rotate labels for better readability
#         for i in range(num_agents, num_features, num_agents):
#             ax.axvline(x=i, color='white', linestyle='-', linewidth=3)
#         ax.invert_yaxis()

#         plt.savefig(f"{hparams.model_dir}saliency_{hparams.model_type}{hparams.window}.png")

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