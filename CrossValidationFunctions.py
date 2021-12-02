import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import itertools
import Constants
import Imageset
import ResNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# implementation of k-fold cross validation
# parameters:
    # 1.) number of splits for k-fold validation
    # 2.) training features
    # 3.) training labels
    # 4.) image transformations for training
    # 5.) image transformations for validations
def k_fold_validation(n_splits, X_train, y_train, training_transform, validation_transform) :
        
    # getting stratified k-folds for equal representation of each class
    stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    # counts f-fold iterations
    
    # placeholders used for evaluation of mean across the cross validation iterations
    iteration = 0
    sum_correct = 0
    sum_total = 0
    evaluations = []
    true_label = np.array([])
    predicted_label = np.array([])
    matrix_sum = torch.zeros(3, 3)

    # running training and validations for each k-fold split    
    for train_indices, validation_indices in stratified_k_fold.split(X_train, y_train.astype('int')):
        
        # merge training features and training labels to get final training and validation sets
        training_data = np.stack((X_train[train_indices], y_train[train_indices]), axis = -1)
        validation_data = np.stack((X_train[validation_indices], y_train[validation_indices]), axis = -1)

        # executing training and collecting individual iteration evaluation results in evaluations list
        evaluations.append(train(iteration, training_data, validation_data, training_transform, testing_transform=validation_transform))
        iteration = iteration + 1
        
    # save k-fold cross validation evaluation results for later redisplay
    np.save(Constants.ROOT_PATH + r'/Processed Dataset/Numpy/CrossValidationEvaluations.npy', evaluations)
    
    # calculations for mean evaluations    
    for evaluation in evaluations:
        sum_correct = sum_correct + evaluation[0]
        sum_total = sum_total + evaluation[1]
        true_label = np.hstack([true_label, evaluation[2]])
        predicted_label = np.hstack([predicted_label, evaluation[3]])
        matrix_sum = matrix_sum + evaluation[4]
    
    # generating mean evaluation report
    print("\nK-Fold Cross Validation Mean of Results")
    print("Accuracy of the model on the test images: {} %". format((sum_correct / sum_total) * 100))
    print("Precision on the test images: {} %".format(100 * precision_score(true_label, predicted_label, average='weighted')))
    print("Recall on the test images: {} %".format(100 * recall_score(true_label, predicted_label, average='weighted')))
    print("F1-Score on the test images: {} %".format(100 * f1_score(true_label, predicted_label, average='weighted')))
    
    # plotting mean confusing matrix
    names = ('Without Mask',
             'With Mask',
             'Other Images')
    
    plt.figure(figsize=(7,7))
    
    # iteration argument set to -2 for mean confusion matrix
    plot_confusion_matrix(-2, matrix_sum/iteration, names)
    plt.show()

# performs training and triggers evaluations
# parameters:
    # 1.) iteration set to -1 for final training and testing, iteration passed from k_fold_validation function for k-fold cross validations
    # 2.) training data (training features and labels merged together)
    # 3.) testing labels (testing features and labels merged together)
    # 4.) image transformations for training
    # 5.) image transformations for testing
def train(iteration, training_data, testing_data, training_transform, testing_transform):

    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate for the model, criterion and optimizer
    model = ResNet.ResNet(ResNet.ResidualBlock, [2, 2, 2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = Constants.LEARNING_RATE)
        
    # performance measurement arrays
    training_loss = []
    training_accuracy = []
    
    # ready image set for training and testing by applying transformations, and creating data loaders
    training_image_set = Imageset.Imageset(training_data, transform = training_transform)
    train_loader = DataLoader(training_image_set, Constants.BATCH_SIZE, shuffle = True)
    
    testing_image_set = Imageset.Imageset(testing_data, transform = testing_transform)
    testing_loader = DataLoader(testing_image_set, Constants.BATCH_SIZE)

    # used for decay of learning rate
    decay = 0

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

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            iterations += 1
        
        training_loss.append(iteration_loss / iterations)
        training_accuracy.append(correct / len(training_image_set))
    
    # save model while running final training
    if iteration == -1:
        torch.save(model.state_dict(), Constants.ROOT_PATH + r'/model.pth.tar')

    # set model to evaluation mode
    model.eval()
    
    # initialize lists for evaluation scores
    test_label = torch.Tensor().to(device)
    predicted_label = torch.Tensor().to(device)
    
    correct = 0
    total = 0
    
    # model testing
    with torch.no_grad():
        
        for images, labels in testing_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # predicted outputs
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
    
            predicted_label = torch.cat((predicted_label, predicted))
            test_label = torch.cat((test_label, labels))
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # generate evaluation report
    evaluation_parameters = evaluation(iteration, training_accuracy, training_loss, predicted_label, test_label, correct, total)

    # return evaluation parameters for calculation of mean of evaluations, useful only for k-fold cross validation
    return evaluation_parameters

# plots confusion matrix
# parameters:
    # 1.) iteration set to -2 for mean confusion matrix, -1 for final testing confusion matrix, and passed from k_fold_validation function for k-fold cross validations
    # 2.) cm is confusion matrix
def plot_confusion_matrix(iteration, cm, classes, cmap=plt.cm.Blues):
    
    if iteration == -2:
        plt.title("Mean Validations Confusion Matrix")
    elif iteration == -1:
        plt.title("Final Testing Confusion Matrix")
    else:
        plt.title("Confusion Matrix for Iteration {}".format(iteration + 1))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    fmt = '.2f' if iteration == -2 else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('Correct label')
    plt.xlabel('Predicted label')

# generates evaluation report
# parameters:
    # 1.) -1 for final testing confusion matrix, and passed from k_fold_validation function for k-fold cross validations
def evaluation(iteration, training_accuracy, training_loss, predicted_label, test_label, correct, total):
    
    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # plot the learning characteristics
    plt.figure(figsize=(8, 8))
    plt.plot(training_accuracy, label = "Training Accuracy")
    plt.plot(training_loss, label = "Training Loss")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')

    if iteration == -1:
        plt.title('Accuracy and Loss Curve for Final Training')
    else:
        plt.title('Accuracy and Loss Curve for Iteration {}'.format(iteration + 1))

    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
    
    # confusion matrix parameters assigned before converting tensors to numpy arrays
    stacked = torch.stack((test_label, predicted_label), dim = 1).int()
    cmt = torch.zeros(3, 3, dtype=torch.int64)
    
    # classification report generation
    if device == Constants.DEVICE_CUDA:
        test_label = test_label.to(Constants.DEVICE_CPU)
        predicted_label = predicted_label.to(Constants.DEVICE_CPU)
    
    test_label = test_label.numpy()
    predicted_label = predicted_label.numpy()
    
    if iteration == -1:
        print("\nFinal Testing Results:")
    else:
        print("\nK-Fold Cross Validation Iteration {}:".format(iteration + 1))
        
    print("Accuracy of the model on the test images: {} %". format((correct / total) * 100))
    print("Precision on the test images: {} %".format(100 * precision_score(test_label, predicted_label, average='weighted')))
    print("Recall on the test images: {} %".format(100 * recall_score(test_label, predicted_label, average='weighted')))
    print("F1-Score on the test images: {} %".format(100 * f1_score(test_label, predicted_label, average='weighted')))
    
    # confusion matrix generation
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    
    names = ('Without Mask',
             'With Mask',
             'Other Images')
    
    plt.figure(figsize=(7,7))
    plot_confusion_matrix(iteration, cmt, names)
    plt.show()
    
    # save final testing evaluation results for later redisplay
    if iteration == -1:
        np.save(Constants.ROOT_PATH + r'/Processed Dataset/Numpy/FinalTestingEvaluations.npy',
                np.array([correct, total, test_label, predicted_label, cmt]))
    
    # returning evaluation parameters to be used for calculation of mean of evaluations, useful only for k-fold cross validation
    return [correct, total, test_label, predicted_label, cmt]