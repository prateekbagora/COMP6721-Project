import os
import numpy as np
import MaskAndNoMask
import Imageset
import Constants
from pathlib import Path
from torch.utils.data.dataloader import DataLoader

# function to retrieve the image dataset
# argument: set 'rebuild_data' to True to process images, recreate and retrieve the dataset 
#               used for first execution and later in case of new additions to dataset
#           set 'rebuild_data' to False to retrieve the dataset from the disk
# returns the image dataset as a numpy array
def get_training_data(rebuild_data):
    if rebuild_data:
        
        # path to raw dataset
        # the organization of folders should be like:
        #   Dataset
        #       With Mask
        #           With Mask 1 (folder consisting of images)
        #           With Mask 2 (folder consisting of images)
        #           and so on ...
        #       Without Mask
        #           Without Mask 1 (folder consisting of images)
        #           Without Mask 2 (folder consisting of images)
        #           and so on ...
        imageset_path = Path(Constants.ROOT_PATH + r'/Dataset')
        
        # path to processed dataset
        destination_path = Path(Constants.ROOT_PATH + r'/Processed Dataset')
        mask_path = imageset_path/'With Mask'
        no_mask_path = imageset_path/'Without Mask'
        Other_images_path = imageset_path/'Other Images'
        
        if not os.path.exists(imageset_path):
            raise Exception("The images' source path '{}' doesn't exist".format(imageset_path))
        if not os.path.exists(destination_path):
            raise Exception("The numpy array's destination path '{}' doesn't exist".format(destination_path))
        
        # directories and the corresponding label
        # 0: Without Mask
        # 1: With Mask
        # 2: Other Images
        imageset_dirs = [ [no_mask_path, 0], [mask_path, 1], [Other_images_path, 2]]
        
        mask_nomask = MaskAndNoMask.MaskAndNoMask(imageset_dirs)
        training_data, _ = mask_nomask.get_training_data()
        
    else:
        try:
            training_data = np.load(Constants.ROOT_PATH + r'/Processed Dataset/Numpy/ImageSet.npy', allow_pickle=True)
        except:
            raise Exception("The numpy array's path '{}' doesn't exist".format(destination_path))
    return training_data

# method to get the mean and standard deviation for the image data set stored as a numpy array
# saves to disk and returns mean and standard deviation
def get_stats(image_array_path):
    
    #load numpy array containing the processed image data set
    training_data = np.load(image_array_path, allow_pickle=True)

    # creating Imageset data set
    image_set = Imageset.Imageset(training_data = training_data)
    mean = 0.0
    std = 0.0
    total_samples = 0.0
    batch_size = 10
    
    # creating data loader for Imageset data set
    image_dataloader = DataLoader(image_set, batch_size)
    
    for data, _ in image_dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    # storing calculated mean and standard deviation in DataSetStats.npy file as a numpy array
    np.save(Constants.ROOT_PATH + r'/Processed Dataset/Numpy/DataSetStats.npy', [mean.numpy(), std.numpy()])
    
    return mean, std