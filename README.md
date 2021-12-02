<p align="justify">Please download the dataset from the kaggle link given below and store it into the Dataset folder of the project.<br>
https://www.kaggle.com/zahmah/face-mask-detector-dataset</p>

## Organization of the Data Set
1. <p align="justify">Place all the folders containing images with mask in 'Dataset/With Mask/' folder.</p>
2. <p align="justify">Place all the folders containing images without mask in 'Dataset/Without Mask/' folder.</p>
3. <p align="justify">Place all the folders containing all other images in 'Dataset/Other Images/' folder.</p>

## Files
### Constants.py
This file contains all the global constants and variables used across the project implementation.
1. ROOT_PATH: set this constant to the complete path of the project folder.
2. Modify hyperparameters as needed.

### MaskAndNoMask.py
<p align="justify">Contains the class MaskAndNoMask which processes the images in the data set and creates a numpy array. Also, saves the array on disk as 'ImageSet.npy'.</p>

### Imageset.py
<p align="justify">Contains the Imageset class which inherits Dataset class, and implements a custom data set which can be used to create data loaders.</p>

### DatasetFunctions.py
1. get_training_data: Enables us to fetch the training data by instantiating MaskAndNoMask class and using its methods.
2. get_stats: Calculates mean and standard deviation for our image dataset, later used for normalization. Also, saves it on disk as a numpy array in file 'DataSetStats.npy' for later use.

### ResNet.py
Contains:
1. conv3x3 function: Creates a convolutional layer with a 3x3 filter.
2. ResidualBlock class: It is the implementation of a single block of the Residual CNN.
3. ResNet class: It is the implementation of the entire Residual CNN.

### TrainingAndTesting.py
<p align="justify">This is where training and testing of the model is implemented.</p>

