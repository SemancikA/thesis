import numpy as np
import nrrd

input_file = '/home/student515/Documents/thesis/Dataset/Lables/Al/0_2_AlSi10Mg-000005-labels.nrrd'

# Read the nrrd file
data, header = nrrd.read(input_file)

# Convert the data to a numpy array
data = np.array(data)

# Print the data
print(data)
print(data.shape)

