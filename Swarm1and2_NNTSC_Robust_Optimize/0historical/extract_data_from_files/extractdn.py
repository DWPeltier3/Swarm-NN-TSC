import os
import re

# Directory containing the text files
directory = '/home/donald.peltier/swarm/logs/decoy_number/Predictions/Predict_10v1C15_Scaled10v10'

# Pattern to match the file names and capture one or two digits
file_pattern = re.compile(r'CNmc20star_10v1C15_Predict_10v(\d{1,2})')

# Lists to hold the extracted values
losses = []
accuracies = []

# Function to extract loss and accuracy from a file
def extract_metrics(filename):
    with open(filename, 'r') as file:
        content = file.read()
        # Extract the metrics
        loss_match = re.search(r'\[([\d.]+), ([\d.]+)\]', content)
        if loss_match:
            loss = float(loss_match.group(1))*100
            accuracy = float(loss_match.group(2))*100
            return loss, accuracy
    return None, None

# List to hold tuples of (decoy number, loss, accuracy)
metrics = []

# Iterate over files in the directory
for filename in os.listdir(directory):
    match = file_pattern.match(filename)
    if match:
        decoy_number = int(match.group(1))
        file_path = os.path.join(directory, filename)
        loss, accuracy = extract_metrics(file_path)
        if loss is not None and accuracy is not None:
            metrics.append((decoy_number, loss, accuracy))

# Sort metrics by decoy number
metrics.sort(key=lambda x: x[0])

# Print the results
print("Loss")
for decoy_number, loss, accuracy in metrics:
    print(f"{decoy_number}. {loss}")
    
print("\nAccuracy")
for decoy_number, loss, accuracy in metrics:
    print(f"{decoy_number}. {accuracy}")

# Write the extracted data for each swarm size to a file
out_filename = f'extracted_data.txt'
output_filename = os.path.join(directory, out_filename)

# Save the results to a file
with open(output_filename, 'w') as summary_file:
    summary_file.write("Loss\n")
    for decoy_number, loss, accuracy in metrics:
        summary_file.write(f"{loss}\n")
    
    summary_file.write("\nAccuracy\n")
    for decoy_number, loss, accuracy in metrics:
        summary_file.write(f"{accuracy}\n")

print(f"Extraction complete. Check '{output_filename}' for the results.")
