import os
import numpy as np

def append_to_csv(data, filename, header):
    # Open the file in append mode
    with open(filename, 'a', newline='') as file:
        file.write(header + "\n") # write variable header for readability
        # Check if the data is a list or an ndarray
        if isinstance(data, list):
            for item in data:
                file.write(str(item) + "\n") # Write each element of the list in a new row
        else:
            # Handle ndarray and Reshape the data if it's multidimensional
            if data.ndim > 1:
                data = data.reshape(-1, data.shape[-1])
            # Use numpy.savetxt to append the array to the file
            np.savetxt(file, data, delimiter=",", header='', comments='')