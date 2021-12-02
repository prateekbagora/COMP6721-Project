The whole dataset which has been gathered for the three classes is in the following repository:
If you want to test the model on whole dataset you can download the dataset from the kaggle link given below, and store it into the Dataset folder of project.

https://www.kaggle.com/zahmah/face-mask-detector-dataset 

Some samples of dataset (1000 images per class) are in the Dataset folder(in the main Project Code folder), its structure is as follow:
	Placed all the folders containing images with mask in '\Dataset\With Mask\' folder
	Placed all the folders containing images without mask in '\Dataset\Without Mask\' folder
	Placed all the folders containing all other images in '\Dataset\Other Images\' folder

If you are executing the project code in Spyder, add PYTHONPATH for the project folder

Files:
	Constants.py:
		This file contains all the global constants and variables used across the project implementation.
		Mandatory changes:
			ROOT_PATH: set this constant to the complete path of the project folder
		Optional changes:
			Modify hyperparameters as needed
		
	MaskAndNoMask.py:
		This file contains the class 'MaskAndNoMask' which processes the images in the data set (essentially, resizing the images to size 200x200 and converting them to 'RGB' format) and creates a numpy array. The 'get_training_data' method performs this task and returns the numpy array and a size dictionary (which gives the number of samples in each class). Also, it saves the array to disk as '\Processed Dataset\Numpy\ImageSet.npy'.
	
	Imageset.py:
		This file contains the 'Imageset' class which inherits Dataset class, and implements a custom data set which will be used to create data loaders for the training and testing data.
	
	DatasetFunctions:
		This file contains two methods:
			get_training_data:
				Enables us to fetch the training data. Pass an argument rebuild_data = True to reprocess the images and rebuild the numpy array '\Processed Dataset\Numpy\ImageSet.npy', by instantiating 'MaskAndNoMask' class and using its method 'get_training_data'. Pass an argument rebuild_data = False to use the existing numpy array '\Processed Dataset\Numpy\ImageSet.npy' from the disk. 
			get_stats:
				Calculates mean and standard deviation for the image dataset, later used for normalization. Also, saves these statistics on disk as a numpy array in file '\Processed Dataset\Numpy\DataSetStats.npy' for later use.
	
	ResNet.py:
		This file contains the implementation of the Residual CNN model. It contains:
			Function 'conv3x3':
				It creates a convolutional layer with a 3x3 filter with the passed input channels, output channels, and stride. It maintains a fixed padding of 1.
			Class 'ResidualBlock':
				It is the implementation of a single residual block of 2 convolutional layers, created using the function 'conv3x3'.
			Class 'ResNet':
				It is the implementation of the final Residual CNN network. The method 'make_layer' creates the residual blocks using the class 'ResidualBlock' and also handles downsampling.
	
	TrainingTestingEvaluation.py:
		Possible changes:
			rebuild_data: Set this variable to False to reprocess the images and regenerate the numpy array data set and data set statistics. Keep it false to use the previously generated files.
		Run this file to perform training and testing, and generate evaluation report.
		This file contains the implementation of training and testing phases, and generation of evaluation reports for the model. It uses the files '\Processed Dataset\Numpy\ImageSet.npy' for the data set, and '\Processed Dataset\Numpy\DataSetStats.npy' to get the data set statistics for normalization. The '\Processed Dataset\Numpy\TestingImageSet.npy' file is generated to enable the rerun of testing later on.
		
	RerunTestingAndEvaluation.py:
		Run this file to perform testing and generate evaluation report.
		This file enables the rerunning of testing phase and generates evaluation results, using '\Processed Dataset\Numpy\TestingImageSet.npy' file.
		
	TrainingAndCrossValidation.py:
This file contains the implementation of 10 fold cross validation on training and generation of evaluation reports for the model. It uses the files '\Processed Dataset\Numpy\ImageSet.npy' for the data set, and '\Processed Dataset\Numpy\DataSetStats.npy' to get the data set statistics for normalization. The '\Processed Dataset\Numpy\FinalTestingEvaluations.npy' file is generated to enable the rerun of testing later on. The '\Processed Dataset\Numpy\CrossValidationEvaluations.npy' file is generated to enable the rerun of 10 fold cross validation later on.

		ComparingEvaluation.py:
	Run this file to perform Final testing and generate evaluation report on each 10 folds of cross validation.
This file enables the rerunning of testing phase and generates evaluation results of validation set, using '\Processed Dataset\Numpy\CrossValidationEvaluations.npy' and '\Processed Dataset\Numpy\FinalTestingEvaluations.npy' file.






Procedure to Run: 
Note: First you need to add the path of project folder to "ROOT_PATH" in "Constants.py" file.

**** Phase 1 ****
 
 1- Open the file "TrainingTestingEvaluation.py" and run it, it will load the data, execute the model and gives the results.

 2. If you want to test or use the trained model again, you have to run "RerunTestingAndEvaluation.py".

**** end Phase 1 ****

**** Phase 2 ****

3. open the "TrainingAndCrossValidation.py" and run it, it will apply the k-fold cross validation and save the evaluation of cross validation on disk as a numpy array in file '\Processed Dataset\Numpy\CrossValidationEvaluations.npy' for later use. Also, it saves the evaluation of testing on disk as a numpy array in file '\Processed Dataset\Numpy\FinalTestingEvaluations.npy' for later use.

4.  If you want to test or use the cross validation model again, you have to run "ComparingEvaluation.py".


**** end Phase 2 ****


Dataset Main Sources:
1- With Mask:
 	https://github.com/cabani/MaskedFace-Net/
2- Without Mask:
	https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset/discussion.	
3- Other Images:
	http://www.vision.caltech.edu/Image_Datasets/Caltech101/
	https://cocodataset.org/#home (fews images)