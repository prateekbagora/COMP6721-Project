import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import Constants
import Imageset
import ResNet
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cuda = torch.device('cuda')
device_cpu = torch.device('cpu')

# setting paths
testing_set_path = Constants.ROOT_PATH + r'/Processed Dataset/Numpy/TestingImageSet.npy'
stats_path = Constants.ROOT_PATH + r'/Processed Dataset/Numpy/DataSetStats.npy'
model_path = Constants.ROOT_PATH + r'/model.pth.tar'

# load precalculated testing data set stored as a numpy array
if not os.path.exists(testing_set_path):
    raise Exception("The testing set source path '{}' doesn't exist".format(testing_set_path))
else:
    testing_data = np.load(testing_set_path, allow_pickle=True)

# load precalculated mean and standard deviation for the data set from disk
if not os.path.exists(stats_path):
    raise Exception("The data set statistics source path '{}' doesn't exist".format(stats_path))
else:
    dataset_mean, dataset_std = np.load(stats_path, allow_pickle = True)

# image transformations
testing_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std)
    ])

# ready image set for testing, and create testing data loader
testing_image_set = Imageset.Imageset(testing_data, transform = testing_transform)
test_loader = DataLoader(testing_image_set, Constants.BATCH_SIZE)

model = ResNet.ResNet(ResNet.ResidualBlock, [2, 2, 2]).to(device)

# loading the trained model
if not os.path.exists(model_path):
    raise Exception("The trained model source path '{}' doesn't exist".format(model_path))
else:
    model.load_state_dict(torch.load(model_path,map_location=device))

# set model to evaluation mode
model.eval()

# initialize lists for evaluation scores
test_label = torch.Tensor().to(device)
predicted_label = torch.Tensor().to(device)

correct = 0
total = 0

# model testing
with torch.no_grad():
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # predicted outputs
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)

        predicted_label = torch.cat((predicted_label, predicted))
        test_label = torch.cat((test_label, labels))
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# confusion matrix parameters assigned before converting tensors to numpy array
stacked = torch.stack((test_label, predicted_label), dim = 1).int()
cmt = torch.zeros(3,3, dtype=torch.int64)

# classification report generation
if device == device_cuda:
    test_label = test_label.to(device_cpu)
    predicted_label = predicted_label.to(device_cpu)

test_label = test_label.numpy()
predicted_label = predicted_label.numpy()

print("\nAccuracy of the model on the test images: {} %". format((correct / total) * 100))
print("Precision on the test images: {} %".format(100 * precision_score(test_label, predicted_label, average='weighted')))
print("Recall on the test images: {} %".format(100 * recall_score(test_label, predicted_label, average='weighted')))
print("F1-Score on the test images: {} %".format(100 * f1_score(test_label, predicted_label, average='weighted')))

# confusion matrix generation
for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix")
    else:
        print('\nConfusion matrix, without normalization')
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('Correct label')
    plt.xlabel('Predicted label')

names = ('Without Mask',
         'With Mask',
         'Other Images')

plt.figure(figsize=(7,7))
plot_confusion_matrix(cmt, names)
plt.show()