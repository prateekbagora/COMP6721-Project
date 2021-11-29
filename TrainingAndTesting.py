import torch
import torch.nn as nn
import sklearn.model_selection as model_selection
import numpy as np
import matplotlib.pyplot as plt
import Constants
import DatasetFunctions
import Imageset
import ResNet
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set rebuild_data to True if you want to reprocess the images and rebuild the training data numpy array
rebuild_data = False

# get all training data
all_data = DatasetFunctions.get_training_data(rebuild_data)

if rebuild_data:
    
    # recalculate training data statistics
    dataset_mean, dataset_std = DatasetFunctions.get_stats(Constants.ROOT_PATH + r'\Processed Dataset\Numpy\Image Set.npy')    

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

# image transformations
training_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(Constants.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(dataset_mean, dataset_std)
    ])

testing_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std)
    ])

# ready image set for training by applying transformations, and create training data loader
training_image_set = Imageset.Imageset(training_data, transform = training_transform)
train_loader = DataLoader(training_image_set, Constants.BATCH_SIZE, shuffle = True)

# ready image set for testing, and create testing data loader
testing_image_set = Imageset.Imageset(testing_data, transform = testing_transform)
test_loader = DataLoader(testing_image_set, Constants.BATCH_SIZE)

# instantiate for the model, criterion and optimizer
model = ResNet.ResNet(ResNet.ResidualBlock, [2, 2, 2]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = Constants.LEARNING_RATE)

# used for decay of learning rate
decay = 0

# performance measurement arrays
training_loss = []
training_accuracy = []

# set model to training mode
model.train()

# model training
for epoch in range(Constants.NUM_EPOCHS):
    
    correct = 0
    iterations = 0
    iteration_loss = 0.0
    
    # decay the learning rate by a factor of DECAY_FACTOR every DECAY_EPOCH_FREQUENCY epochs
    if (epoch + 1) % Constants.DECAY_EPOCH_FREQUENCY == 0:
        decay += 1
        optimizer.param_groups[0]['lr'] = Constants.LEARNING_RATE * (Constants.DECAY_FACTOR ** decay)
        print("The new learning rate is {}".format(optimizer.param_groups[0]['lr']))
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        iteration_loss += loss.item()
        
        # backpropogation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, Constants.NUM_EPOCHS, i + 1, len(train_loader), loss.item()))
            
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        iterations += 1
    
    training_loss.append(iteration_loss / iterations)
    training_accuracy.append(correct / len(training_image_set))

# set model to evaluation mode
model.eval()

# model testing
with torch.no_grad():
    
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # predicted outputs
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print("Accuracy of the model on the test images: {} %". format((correct / total) * 100))

# save model
torch.save(model.state_dict(), Constants.ROOT_PATH + r'\model.pth.tar')

# plot the evaluation results
f = plt.figure(figsize=(10, 10))
plt.plot(training_accuracy, label = "Training Accuracy")
plt.plot(training_loss, label = "Training Loss")
plt.legend()
plt.show()