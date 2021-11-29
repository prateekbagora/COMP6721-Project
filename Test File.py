import torch
import sklearn.model_selection as model_selection
import numpy as np
import Constants
import DatasetFunctions


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set rebuild_data to True if you want to reprocess the images and rebuild the training data numpy array
rebuild_data = True

# get all training data
all_data = DatasetFunctions.get_training_data(rebuild_data)

if rebuild_data:
    
    # recalculate training data statistics
    dataset_mean, dataset_std = DatasetFunctions.get_stats(Constants.ROOT_PATH + r'\Processed Dataset\Numpy\ImageSet.npy')    

else:
    
    # load precalculated mean and standard deviation for the data set from disk
    dataset_mean, dataset_std = np.load(Constants.ROOT_PATH + r'\Processed Dataset\Numpy\DataSetStats.npy', allow_pickle = True)
    
# split images and labels
all_data_images = all_data[:, 0]
all_data_labels = all_data[:, 1]

# split the data set into training and testing data sets with equal distribution of classes
X_train, X_test, y_train, y_test = model_selection.train_test_split(all_data_images, all_data_labels,
                                                                    test_size = Constants.TRAIN_TEST_RATIO, random_state = 1,
                                                                    stratify = all_data_labels)

# merge again to get final training and testing sets
training_data = np.stack((X_train, y_train), axis = -1)
testing_data = np.stack((X_test, y_test), axis = -1)