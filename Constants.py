import torch

# set this to the location of the project folder
ROOT_PATH = r'.'

# set this to the image size you need
# all images will be resized to this during preprocessing
# change the size of the max pool layer of ResNet accordingly
IMG_SIZE = 200

# training hyperparameters
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
BATCH_SIZE = 20
TRAIN_TEST_RATIO = 0.2
DECAY_FACTOR = 0.5
DECAY_EPOCH_FREQUENCY = 10
DEVICE_CUDA = torch.device('cuda')
DEVICE_CPU = torch.device('cpu')
N_SPLITS = 10