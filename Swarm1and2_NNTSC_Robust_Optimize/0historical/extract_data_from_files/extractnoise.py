import re
import os

# The directory where your text files are stored
directory = '/home/donald.peltier/swarm/logs/historical/noise/compare/predict_CNmhFULL_noise_0_combscaled'  # Replace with the path to your files

# Patterns to match the desired numbers
# val_loss_pattern = r"Minimum Val Loss: ([0-9.]+)" % used for original paper (validation loss)
test_loss_pattern = r"model.metrics_names:\s*\[.*?\]\n\[\s*([0-9.]+)" # when comparing test set loss
metrics_pattern = r"model.metrics_names:\s*\[.*?\]\n\[(.*?)\]"
f1_score_pattern = r"(\bCOMMS\b|\bPRONAV\b)\s+\d\.\d+\s+\d\.\d+\s+(\d\.\d+)"
confusion_matrix_pattern = r"Label(\d) (COMMS|PRONAV)\s+\[\[\s*(\d+)\s+(\d+)\s*\]\s*\[\s*(\d+)\s+(\d+)\s*\]\]"

# Lists to store the extracted numbers
losses = []
attr_acc = []
class_acc = []
comms_f1_scores = []
pronav_f1_scores = []
comms_confusion_matrices = []
pronav_confusion_matrices = []

# Loop over the files in the directory
for i in range(0, 51):
    filename = f"predict_noise_{i}.txt"
    filepath = os.path.join(directory, filename)

    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            content = file.read()

            # Extract 'Minimum Val Loss'
            loss_matches = re.findall(test_loss_pattern, content)
            if loss_matches:
                loss = int(round(float(loss_matches[0]) * 100))
                losses.append(loss)
            else:
                losses.append('N/A')

            # Extract accuracy values
            metrics_matches = re.findall(metrics_pattern, content, re.MULTILINE)
            for match in metrics_matches:
                numbers = [x.strip() for x in match.split(',')]
                if len(numbers) >= 5:
                    class_acc.append(int(round(float(numbers[3]) * 100)))  # 4th number
                    attr_acc.append(int(round(float(numbers[4]) * 100)))  # 5th number
                else:
                    class_acc.append('N/A')
                    attr_acc.append('N/A')

            # Extract f1-scores for COMMS and PRONAV
            f1_scores = re.findall(f1_score_pattern, content)
            comms_f1 = 'N/A'
            pronav_f1 = 'N/A'
            for match in f1_scores:
                if match[0] == 'COMMS':
                    comms_f1 = int(round(float(match[1]) * 100))
                elif match[0] == 'PRONAV':
                    pronav_f1 = int(round(float(match[1]) * 100))
            comms_f1_scores.append(comms_f1)
            pronav_f1_scores.append(pronav_f1)

            # Extract confusion matrix values for both COMMS and PRONAV
            confusion_matrix_matches = re.findall(confusion_matrix_pattern, content)
            comms_cm = ('N/A', 'N/A', 'N/A', 'N/A')
            pronav_cm = ('N/A', 'N/A', 'N/A', 'N/A')
            for match in confusion_matrix_matches:
                label, matrix_label, tn, fp, fn, tp = match
                if matrix_label == 'COMMS':
                    comms_cm = (int(tn), int(fp), int(fn), int(tp))
                elif matrix_label == 'PRONAV':
                    pronav_cm = (int(tn), int(fp), int(fn), int(tp))
            comms_confusion_matrices.append(comms_cm)
            pronav_confusion_matrices.append(pronav_cm)

    else:
        losses.append('N/A')
        class_acc.append('N/A')
        attr_acc.append('N/A')
        comms_f1_scores.append('N/A')
        pronav_f1_scores.append('N/A')
        comms_confusion_matrices.append(('N/A', 'N/A', 'N/A', 'N/A'))
        pronav_confusion_matrices.append(('N/A', 'N/A', 'N/A', 'N/A'))
        print(f"File not found: {filename}")

# The directory where you want to save the extracted_data.txt file
out_filename = f'extracted_data.txt'
output_filename = os.path.join(directory, out_filename)

# Write the extracted and rounded numbers to a new text file with headers
with open(output_filename, 'w') as output_file:
    output_file.write('Test Loss,Class Accuracy,Attr Accuracy\n')  # Writing headers
    for i in range(len(losses)):  # Assuming all lists are of the same length
        output_file.write(f"{losses[i]},{class_acc[i]},{attr_acc[i]}\n")
    
    output_file.write('\n\nCOMMS f1-score,PRONAV f1-score\n')  # Writing new headers
    for i in range(len(comms_f1_scores)):  # Assuming all lists are of the same length
        output_file.write(f"{comms_f1_scores[i]},{pronav_f1_scores[i]}\n")

    # Writing headers and data for COMMS confusion matrix
    output_file.write('\n\nCOMMS TN,FP,FN,TP\n')
    for cm in comms_confusion_matrices:
        output_file.write(f"{cm[0]},{cm[1]},{cm[2]},{cm[3]}\n")

    # Writing headers and data for PRONAV confusion matrix
    output_file.write('\n\nPRONAV TN,FP,FN,TP\n')
    for cm in pronav_confusion_matrices:
        output_file.write(f"{cm[0]},{cm[1]},{cm[2]},{cm[3]}\n")

print(f"Extraction complete. Check '{output_filename}' for the results.")
