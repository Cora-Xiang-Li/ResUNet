import h5py
import numpy as np
import matplotlib.pyplot as plt

# Open the HDF5 file
hdf5_file = '/home/xiangli/ResUNet/data/VGG/VGG.hdf5'
with h5py.File(hdf5_file, 'r') as f:
    # Access the density maps dataset
    density_maps = f['counts']  # Replace 'density_maps' with the actual name of the dataset
    
    # Print the shape of the dataset
    print("Shape of the dataset:", density_maps.shape)
    
    # Access the first density map
    first_density_map = density_maps[0]
    
    # Plot and save the first density map
    plt.imshow(first_density_map, cmap='hot', interpolation='nearest')
    plt.title('First Density Map')
    plt.colorbar()
    plt.savefig('first_density_map.png')  # Save the image
    plt.show()
    
    # Print the range of values in the HDF5 file
    data_range = (np.min(density_maps), np.max(density_maps))
    print(f"Range of values in the HDF5 file: {data_range}")
    
print("First density map saved as 'first_density_map.png'")
