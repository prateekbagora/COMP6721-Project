## Organization of the Data Set
1. <p align="justify">Place all the folders containing images with mask in 'Dataset/With Mask/' folder</p>
2. <p align="justify">Place all the folders containing images without mask in 'Dataset/Without Mask/' folder</p>
3. <p align="justify">Place all the folders containing all other images in 'Dataset/Other Images/' folder</p>

## Files
### Constants.py
1. ROOT_PATH: set this constant to the complete path of the project folder
2. Modify hyperparameters as needed

### MaskAndNoMask.py
<p align="justify">Contains the class MaskAndNoMask which processes the images in the data set and creates a numpy array. Also, saves the array on disk as 'ImageSet.npy'</p>

### Imageset.py
<p align="justify">Contains the Imageset class which inherits Dataset class, and implements a custom data set which can be used to create data loaders</p>

### DatasetFunctions.py
1. get_training_data:
<p align="justify">Enables us to fetch the training data by instantiating MaskAndNoMask class and using its methods
2. get_stats:
<p align="justify">Calculates mean and standard deviation for our image dataset, later used for normalization. Also, saves it on disk as a numpy array in file
