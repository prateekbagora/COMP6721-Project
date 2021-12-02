import torch
import numpy as np
import pandas as pd
import Constants
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import itertools
import math

def plot_confusion_matrix(iteration, cm, classes, axes, row, col, cmap=plt.cm.Blues):
        
    axes[row,col].imshow(cm, interpolation='nearest', cmap=cmap)

    fmt = '.2f' if iteration == -2 else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[row,col].text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    for ax in axes.flat:
        ax.set(xlabel='Predicted label', ylabel='Correct label')

    plt.setp(axes, xticks=np.arange(len(classes)),xticklabels=classes)  
    plt.setp(axes, yticks=np.arange(len(classes)),yticklabels=classes)  
    
    for ax in axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            
    if iteration == -2:
        axes[row, col].set_title("Mean Validations Confusion Matrix")
    elif iteration == -1:
        axes[row, col].set_title("Final Testing Confusion Matrix")
    else:
        axes[row, col].set_title("Confusion Matrix for Iteration {}".format(iteration + 1))            

    plt.tight_layout()

iteration = 0
sum_correct = 0
sum_total = 0
true_label = np.array([])
predicted_label = np.array([])
matrix_sum = torch.zeros(3, 3)
metrics = []
confusion=[]

# retrieve the evaluation results saved on the disk
kfold_evaluations = np.load(Constants.ROOT_PATH + r'/Processed Dataset/Numpy/CrossValidationEvaluations.npy', allow_pickle=True)
testing_evaluations = np.load(Constants.ROOT_PATH + r'/Processed Dataset/Numpy/FinalTestingEvaluations.npy', allow_pickle=True)

# create a single list of all the evaluation metrics
for index, evaluation in enumerate(kfold_evaluations):
    
    iteration_metric = [(evaluation[0]/evaluation[1]) * 100,
                             100 * precision_score(evaluation[2], evaluation[3], average='weighted'),
                             100 * recall_score(evaluation[2], evaluation[3], average='weighted'),
                             100 * f1_score(evaluation[2], evaluation[3], average='weighted')]
    metrics.append(iteration_metric)
    confusion.append([index, (evaluation[4])])

    # calculating mean results of all the cross validation iterations    
    sum_correct = sum_correct + evaluation[0]
    sum_total = sum_total + evaluation[1]
    true_label = np.hstack([true_label, evaluation[2]])
    predicted_label = np.hstack([predicted_label, evaluation[3]])
    matrix_sum = matrix_sum + evaluation[4]
    iteration = iteration + 1
    
# adding mean results of the k-fold cross validation to evaluation metrics list
iteration_metric = [(sum_correct/sum_total) * 100,
                    100 * precision_score(true_label, predicted_label, average='weighted'),
                    100 * recall_score(true_label, predicted_label, average='weighted'),
                    100 * f1_score(true_label, predicted_label, average='weighted')]
metrics.append(iteration_metric)
confusion.append([-2, matrix_sum/iteration])

# adding final testing results to evaluation metrics list
iteration_metric = [(testing_evaluations[0]/testing_evaluations[1]) * 100,
                    100 * precision_score(testing_evaluations[2], testing_evaluations[3], average='weighted'),
                    100 * recall_score(testing_evaluations[2], testing_evaluations[3], average='weighted'),
                    100 * f1_score(testing_evaluations[2], testing_evaluations[3], average='weighted')]
metrics.append(iteration_metric)
confusion.append([-1, testing_evaluations[4]])

# generating index for data frame
index = []
for iteration in range(len(kfold_evaluations)):
    index.append("Iteration {}".format(iteration + 1))  
index.extend(["K-Fold Mean", "Final Testing"])

# generating columns for data frame
columns = ["Accuracy", "Precision", "Recall", "F1-Score"]

# creating data frame with all the evaluation metrics
metrics = pd.DataFrame(metrics, index, columns)
for col in columns:
    metrics[col] = metrics[col].map('{:.2f}%'.format)

print(metrics,"\n\n")

total_figures = Constants.N_SPLITS + 2
nrows = (total_figures / 3) if (total_figures / 3) == 0 else math.floor(total_figures / 3) + 1
fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15,15))

names = ('Without Mask',
         'With Mask',
         'Other Images')

# call confusion matrix for each of iteration in k-fold evaluation, mean evaluation and final testing
for (x_axis, y_axis), matrix in zip(itertools.product(np.arange(axes.shape[0]), np.arange(axes.shape[1])), confusion):
    plot_confusion_matrix(matrix[0], matrix[1], names, axes, x_axis, y_axis)

# hide blank subplots
blank_axes = (nrows * 3) - total_figures
column = -1
for axis in range(blank_axes):
    axes[-1][column].set_visible(False)
    column = column - 1