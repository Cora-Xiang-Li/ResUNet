import os
import h5py
import numpy as np
from PIL import Image
import os
import h5py
import numpy as np

def read_h5_file(filepath):
    with h5py.File(filepath, 'r') as hdf:    
        coordinates = hdf['coordinates'][:]        
        return coordinates

def sort_dots(fold_path, h5_file):
# /home/xiangli/ResUNet/data/BCData/annotations/train/positive/0.h5
    pos_filepath = os.path.join(fold_path, 'positive', h5_file)
    neg_filepath = os.path.join(fold_path, 'negative', h5_file)

    pos_coords = read_h5_file(pos_filepath)
    neg_coords = read_h5_file(neg_filepath)

    i, j = 0, 0
    single_img_dots = []
    while len(single_img_dots) < len(pos_coords)+len(neg_coords):
        if i == len(pos_coords):
            for j1 in neg_coords[j:]:
                single_img_dots.append(j1)
            break
        elif j == len(neg_coords):
            for i1 in pos_coords[i:]:
                single_img_dots.append(i1)
            break
        else:
            if pos_coords[i][1] <= neg_coords[j][1]:
                single_img_dots.append(pos_coords[i])
                i+=1
            else:
                single_img_dots.append(neg_coords[j])
                j+=1
    print(len(single_img_dots))
    return single_img_dots
 
def read_dots(fold_path, h5_file):
    pos_filepath = os.path.join(fold_path, 'positive', h5_file)
    neg_filepath = os.path.join(fold_path, 'negative', h5_file)
    all_dots = []
    pos_coords = read_h5_file(pos_filepath)
    neg_coords = read_h5_file(neg_filepath)
    if len(pos_coords) != 0 and len(neg_coords) != 0:
        all_dots= np.concatenate((pos_coords, neg_coords), axis=0)
    elif len(pos_coords) != 0 and len(neg_coords) == 0:
        all_dots = pos_coords
    elif len(pos_coords) == 0 and len(neg_coords) != 0:
        all_dots = neg_coords
    else:
        return np.zeros((640, 640), dtype=int)

    # Initialize the array with zeros
    single_img_dots = []
    array_size = 640
    single_img_dots = np.zeros((array_size, array_size), dtype=int)

    for coord in all_dots:
        x, y = coord
        single_img_dots[y, x] = 1

    return single_img_dots

def load_data(folders):
    images = []
    dots = []
    for folder in folders:
        for filename in sorted(os.listdir(folder)):
            if filename.endswith('.png'):
                img = Image.open(os.path.join(folder, filename))
                # img = img.convert('L')  # Convert to grayscale
                img = np.array(img)
                images.append(img)
                
                # Get annotation
                annotations_dir = '/home/xiangli/ResUNet/data/BCData/annotations'
                base_name = os.path.basename(folder)
                fold_path = os.path.join(annotations_dir, base_name)
                h5_file = filename.replace('.png', '.h5')
                dot = read_dots(fold_path, h5_file)
                dots.append(dot)

    return images, dots

image_folders = [
    '/home/xiangli/ResUNet/data/BCData/images/train',
    '/home/xiangli/ResUNet/data/BCData/images/test',
    '/home/xiangli/ResUNet/data/BCData/images/validation'
    ]

images, dots = load_data(image_folders)
output_filepath = '/home/xiangli/ResUNet/data/BCD/BCD.hdf5'
with h5py.File(output_filepath, 'w') as f:
        f.create_dataset('imgs', data=np.array(images), compression='gzip', compression_opts=9)
        f.create_dataset('counts', data=np.array(dots), compression='gzip', compression_opts=9)
        print("images and annotations have been saved to a single HDF5 file.")
