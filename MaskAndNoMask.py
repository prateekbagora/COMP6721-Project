import os
import numpy as np
import Constants
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

class MaskAndNoMask():

    # processed image size
    LABELS = {'Without Mask': 0, 'With Mask': 1}

    def __init__(self, imageset_dirs):
        
        # image set directories passed in the form of an array
        # example: [[no_mask_path, 0], [mask_path, 1]]
        #   where the first element is a directory path and the second element is a label in each sequence
        self.imageset_dirs = imageset_dirs
        
        # dictionary to hold the number of samples for each class
        self.imageset_size = {'Without Mask': 0, 'With Mask': 0}
        
        # numpy array to hold the processed image dataset
        self.training_data = []

    # method that processes image dataset and stores it in a numpy array
    def get_training_data(self):
        for imageset_dir, label in self.imageset_dirs:
            for folder in tqdm(os.listdir(imageset_dir)):
                folder_path = os.path.join(imageset_dir, folder)
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    try:
                        transform = transforms.Compose([transforms.Resize([Constants.IMG_SIZE, Constants.IMG_SIZE]), ])
                        img = Image.open(image_path).convert('RGB')
                        img = transform(img)
                        self.training_data.append([np.array(img), label])
                        if label == 1:
                            self.imageset_size['With Mask'] += 1
                        if label == 0:
                            self.imageset_size['Without Mask'] += 1
                    except:
                        raise Exception('Error: {}'.format(image_path))
        np.random.shuffle(self.training_data)
        np.save(Constants.ROOT_PATH + r'\Processed Dataset\Numpy\ImageSet.npy', self.training_data)
        return np.array(self.training_data), self.imageset_size