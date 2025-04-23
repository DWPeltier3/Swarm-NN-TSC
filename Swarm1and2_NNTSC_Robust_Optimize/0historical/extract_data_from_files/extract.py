import re
import os

num_slurm_jobs=24
# Define model types, output lengths, swarm sizes, and window types
# model_types = ["cn", "fcn", "tr", "tr"]
# output_lengths = ["vec", "vec", "vec", "seq"]
model_types = ["lstm"]
output_lengths = ["vec"]
swarm_sizes = [25, 50, 75, 100]
# swarm_sizes = [25]
# window_types = [20, -1]
window_types = [-1]

# The directory where your text files are stored
directory = '/home/donald.peltier/swarm/logs'  # Update as needed

# Patterns to match the desired numbers
val_loss_pattern = r"Minimum Val Loss: ([0-9.]+)"
metrics_pattern = r"model.metrics_names:\s*\[.*?\]\n\[(.*?)\]"
f1_score_pattern = r"(\bCOMMS\b|\bPRONAV\b)\s+\d\.\d+\s+\d\.\d+\s+(\d\.\d+)"
info_pattern = r"data_path\s+:\s+/home/donald\.peltier/swarm/data/data_(\d+)v\d+_r\d+s_.*\.npz\s+window\s+:\s+(-?\d+)\s+.*\s+model_type\s+:\s+(\w+)\s+.*\s+output_length\s+:\s+(\w+)"

# The directory where you want to save the extracted_data.txt files
output_directory = '/home/donald.peltier/swarm/logs/extracted_data'  # Update as needed

for swarm_size in swarm_sizes:
    all_data = []
    
    for model_type, output_length in zip(model_types, output_lengths):
        for window_type in window_types:
            data = {'model_type': model_type, 'output_length': output_length, 'window_type': window_type}
            val_losses = []
            class_acc = []
            attr_acc = []
            comms_f1_scores = []
            pronav_f1_scores = []

            # Loop over the files in the directory for the current combination
            for i in range(1, num_slurm_jobs+1):
                filename = f"swarm-class_{i}.txt"
                filepath = os.path.join(directory, filename)

                if os.path.isfile(filepath):
                    with open(filepath, 'r') as file:
                        content = file.read()

                        # Extract model info
                        info_matches = re.search(info_pattern, content)
                        if info_matches and \
                           int(info_matches.group(1)) == swarm_size and \
                           int(info_matches.group(2)) == window_type and \
                           info_matches.group(3) == model_type and \
                           info_matches.group(4) == output_length:
                            # Extract 'Minimum Val Loss'
                            val_loss_matches = re.findall(val_loss_pattern, content)
                            if val_loss_matches:
                                val_loss = int(round(float(val_loss_matches[0]) * 100))
                                val_losses.append(val_loss)
                            else:
                                val_losses.append('N/A')

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

            # Append the extracted data to the all_data list
            data['val_losses'] = val_losses
            data['class_acc'] = class_acc
            data['attr_acc'] = attr_acc
            data['comms_f1_scores'] = comms_f1_scores
            data['pronav_f1_scores'] = pronav_f1_scores
            all_data.append(data)

    # Write the extracted data for each swarm size to a file
    out_filename = f'extracted_data_swarm{swarm_size}.txt'
    output_filename = os.path.join(output_directory, out_filename)

    with open(output_filename, 'w') as output_file:
        for data in all_data:
            output_file.write(f"{data['model_type']} {data['output_length']} {data['window_type']}\n")
            output_file.write('Val Loss,Class Accuracy,Attr Accuracy\n')
            for i in range(len(data['val_losses'])):
                output_file.write(f"{data['val_losses'][i]},{data['class_acc'][i]},{data['attr_acc'][i]}\n")
            output_file.write('\nCOMMS f1-score,PRONAV f1-score\n')
            for i in range(len(data['comms_f1_scores'])):
                output_file.write(f"{data['comms_f1_scores'][i]},{data['pronav_f1_scores'][i]}\n")
            output_file.write('\n')

    print(f"Extraction complete for swarm size {swarm_size}. Check '{output_filename}' for the results.")
