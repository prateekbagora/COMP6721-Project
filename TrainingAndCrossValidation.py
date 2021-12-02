import torch
import numpy as np
import Constants
import DatasetFunctions
import CrossValidationFunctions
from torchvision import transforms
from sklearn import model_selection

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set rebuild_data to True if you want to reprocess the images and rebuild the training data numpy array
rebuild_data = True

# get all training data
all_data = DatasetFunctions.get_training_data(rebuild_data)

if rebuild_data:
    
    # recalculate training data statistics
    dataset_mean, dataset_std = DatasetFunctions.get_stats(Constants.ROOT_PATH + r'/Processed Dataset/Numpy/ImageSet.npy')    

else:
    
    # load precalculated mean and standard deviation for the data set from disk
    dataset_mean, dataset_std = np.load(Constants.ROOT_PATH + r'/Processed Dataset/Numpy/DataSetStats.npy', allow_pickle = True)
    
# split images and labels
all_data_images = all_data[:, 0]
all_data_labels = all_data[:, 1]

# split the data set into training and testing data sets with equal distribution of classes
X_train, X_test, y_train, y_test = model_selection.train_test_split(all_data_images, all_data_labels,
                                                                    test_size = Constants.TRAIN_TEST_RATIO, random_state = 1,
                                                                    stratify = all_data_labels)

# merge again to get final testing set
testing_data = np.stack((X_test, y_test), axis = -1)

# save testing data numpy array to the disk to be later used to rerun testing and generating evaluation report
np.save(Constants.ROOT_PATH + r'/Processed Dataset/Numpy/TestingImageSet.npy', testing_data)

# image transformations for training
training_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(Constants.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(dataset_mean, dataset_std)
    ])

# image transformations for testing
testing_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std)
    ])

# run k-fold validations
# we need to pass the following parameters:
    # 1.) number of splits for k-fold validation
    # 2.) training features
    # 3.) training labels
    # 4.) image transformations for training
    # 5.) image transformations for validations
CrossValidationFunctions.k_fold_validation(Constants.N_SPLITS, X_train, y_train, training_transform, validation_transform=testing_transform)

# merge again to get final training set
training_data = np.stack((X_train, y_train), axis = -1)

# run final training and testing
# we need to pass the following parameters:
    # 1.) iteration set to -1 for final training and testing
    # 2.) training data (training features and labels merged together)
    # 3.) testing labels (testing features and labels merged together)
    # 4.) image transformations for training
    # 5.) image transformations for testing
CrossValidationFunctions.train(-1, training_data, testing_data, training_transform, testing_transform)