## Organization of the Data Set
1. <p align="justify">Place all the folders containing images with mask in 'Dataset/With Mask/' folder</p>
2. <p align="justify">Place all the folders containing images without mask in 'Dataset/Without Mask/' folder</p>
3. <p align="justify">Place all the folders containing all other images in 'Dataset/Other Images/' folder</p>

## Files
### Constants.py
1. ROOT_PATH: set this constant to the complete path of the project folder
2. Modify hyperparameters as needed

### MaskAndNoMask.py
Contains the class MaskAndNoMask which processes the images in the data set and creates a numpy array. Also, saves the array on disk as 'ImageSet.npy'

### Imageset.py
Contains the Imageset class which inherits Dataset class, and implements a custom data set which can be used to create data loaders

### DatasetFunctions.py
#### get_training_data:
Enables us to fetch the training data by instantiating MaskAndNoMask class and using its methods
#### get_stats:
Calculates mean and standard deviation for our image dataset, later used for normalization. Also, saves it on disk as a numpy array in file
