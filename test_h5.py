import h5py
import numpy as np

def print_attrs(name, obj):
    """Helper function to print the attributes of a given object"""
    print(f"\nName: {name}")
    print(f"Type: {'Group' if isinstance(obj, h5py.Group) else 'Dataset'}")
    print("Attributes:")
    for key, val in obj.attrs.items():
        print(f"  {key}: {val}")
    if isinstance(obj, h5py.Dataset):
        print(f"Shape: {obj.shape}")
        print(obj[0])
        print(f"Data Type: {obj.dtype}")

# Replace 'example.h5' with the path to your HDF5 file
file_path = '/home/xiangli/ResUNet/data/VGG/VGG.hdf5'
# file_path = '/home/xiangli/ResUNet/data/BCData/annotations/test/negative/0.h5'
# file_path = '/home/xiangli/ResUNet/data/pseudo_labels/pseudo_labels.hdf5'
# Open the HDF5 file in read mode
with h5py.File(file_path, 'r') as hdf:
    # List all groups and datasets in the file
    # def print_attrs(name, obj):
    #     print(name)
    #     for key, val in obj.attrs.items():
    #         print(f"    {key}: {val}")
    # def print_structure(name, obj):
    #             if isinstance(obj, h5py.Dataset):
    #                 print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
    #             elif isinstance(obj, h5py.Group):
    #                 print(f"Group: {name}")
                    
    hdf.visititems(print_attrs)
    
    # Assume 'coordinates' dataset exists, containing the annotations
    coordinates = hdf['imgs'][:]
    
    # Print the shape and type of the dataset
    print("Coordinates dataset shape:", coordinates.shape)
    print("Coordinates dataset data type:", coordinates.dtype)
    
    # Get the range of x and y coordinates
    # x_coords = coordinates[:, 0]
    # y_coords = coordinates[:, 1]
    
    # print("X coordinates range:", x_coords.min(), "to", x_coords.max())
    # print("Y coordinates range:", y_coords.min(), "to", y_coords.max())

    # # Optionally, print some of the coordinates
    # print("Sample coordinates:", coordinates[:10])
