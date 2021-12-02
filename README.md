<p align="justify">Please download the dataset from the kaggle link given below and store it into the Dataset folder of the project.<br>
https://www.kaggle.com/zahmah/face-mask-detector-dataset</p>

## Organization of the Data Set
1. <p align="justify">Place all the folders containing images with mask in 'Dataset/With Mask/' folder.</p>
2. <p align="justify">Place all the folders containing images without mask in 'Dataset/Without Mask/' folder.</p>
3. <p align="justify">Place all the folders containing all other images in 'Dataset/Other Images/' folder.</p>

## Files
### Constants.py
<p align="justify">This file contains all the global constants and variables used across the project implementation.</p>
1. ROOT_PATH: set this constant to the complete path of the project folder.
2. Modify hyperparameters as needed.

### MaskAndNoMask.py
<p align="justify">This file contains the class 'MaskAndNoMask' which processes the images in the data set (essentially, resizing the images to size 200x200 and converting them to 'RGB' format) and creates a numpy array. The 'get_training_data' method performs this task and returns the numpy array and a size dictionary (which gives the number of samples in each class). Also, it saves the array to disk as '\Processed Dataset\Numpy\ImageSet.npy'.</p>

### Imageset.py
<p align="justify">This file contains the 'Imageset' class which inherits Dataset class, and implements a custom data set which will be used to create data loaders for the training and testing data.</p>

### DatasetFunctions.py
This file contains two methods:
1. get_training_data: Enables us to fetch the training data. Pass an argument rebuild_data = True to reprocess the images and rebuild the numpy array '\Processed Dataset\Numpy\ImageSet.npy', by instantiating 'MaskAndNoMask' class and using its method 'get_training_data'. Pass an argument rebuild_data = False to use the existing numpy array '\Processed Dataset\Numpy\ImageSet.npy' from the disk.
2. get_stats: Calculates mean and standard deviation for the image dataset, later used for normalization. Also, saves these statistics on disk as a numpy array in file '\Processed Dataset\Numpy\DataSetStats.npy' for later use.

### ResNet.py
This file contains the implementation of the Residual CNN model. It contains:
1. conv3x3 function: It creates a convolutional layer with a 3x3 filter with the passed input channels, output channels, and stride. It maintains a fixed padding of 1.
2. ResidualBlock class: It is the implementation of a single residual block of 2 convolutional layers, created using the function 'conv3x3'.
3. ResNet class: It is the implementation of the final Residual CNN network. The method 'make_layer' creates the residual blocks using the class 'ResidualBlock' and also handles downsampling.

### TrainingTestingEvaluation.py:
<p align="justify">rebuild_data: Set this variable to False to reprocess the images and regenerate the numpy array data set and data set statistics. Keep it false to use the previously generated files.</p>
<p align="justify">Run this file to perform training and testing, and generate evaluation report.</p>
<p align="justify">This file contains the implementation of training and testing phases, and generation of evaluation reports for the model. It uses the files '\Processed Dataset\Numpy\ImageSet.npy' for the data set, and '\Processed Dataset\Numpy\DataSetStats.npy' to get the data set statistics for normalization. The '\Processed Dataset\Numpy\TestingImageSet.npy' file is generated to enable the rerun of testing later on.</p>

### RerunTestingAndEvaluation.py:
<p align="justify">Run this file to perform testing and generate evaluation report.</p>
<p align="justify">This file enables the rerunning of testing phase and generates evaluation results, using '\Processed Dataset\Numpy\TestingImageSet.npy' file.</p>

### TrainingAndCrossValidation.py:
<p align="justify">This file contains the implementation of 10 fold cross validation on training and generation of evaluation reports for the model. It uses the files '\Processed Dataset\Numpy\ImageSet.npy' for the data set, and '\Processed Dataset\Numpy\DataSetStats.npy' to get the data set statistics for normalization. The '\Processed Dataset\Numpy\FinalTestingEvaluations.npy' file is generated to enable the rerun of testing later on. The '\Processed Dataset\Numpy\CrossValidationEvaluations.npy' file is generated to enable the rerun of 10 fold cross validation later on.</p>

### ComparingEvaluation.py:
<p align="justify">Run this file to perform Final testing and generate evaluation report on each 10 folds of cross validation.
This file enables the rerunning of testing phase and generates evaluation results of validation set, using '\Processed Dataset\Numpy\CrossValidationEvaluations.npy' and '\Processed Dataset\Numpy\FinalTestingEvaluations.npy' file.</p>

## Procedure to Run:
Note: First you need to add the path of project folder to "ROOT_PATH" in "Constants.py" file.

**** Phase 1 ****
 
 1- Open the file "TrainingTestingEvaluation.py" and run it, it will load the data, execute the model and gives the results.

 2. If you want to test or use the trained model again, you have to run "RerunTestingAndEvaluation.py".

**** end Phase 1 ****

**** Phase 2 ****

3. open the "TrainingAndCrossValidation.py" and run it, it will apply the k-fold cross validation and save the evaluation of cross validation on disk as a numpy array in file '\Processed Dataset\Numpy\CrossValidationEvaluations.npy' for later use. Also, it saves the evaluation of testing on disk as a numpy array in file '\Processed Dataset\Numpy\FinalTestingEvaluations.npy' for later use.

4.  If you want to test or use the cross validation model again, you have to run "ComparingEvaluation.py".


**** end Phase 2 ****
