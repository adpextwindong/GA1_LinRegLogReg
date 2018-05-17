"""Usage:
	cifar_p3.py <depth> <activation> <epochs> <learning_rate> <dropout> <momentum> <weight_decay> [--c|--cont] [-l|--logging]

    Arguments:
        depth: INT - 1 as Single or a value > 1 for multilayer
        activation: SIG - Sigmoid Function
            RELU - ReLu Activation Function
        epochs: INTEGER
        learning_rate: FLOAT
        dropout: FLOAT
        momentum: FLOAT
        weight_decay: FLOAT

    Options:
        -h --help     Show this screen.
        -c --cont     Save model to resume later. Will resume stored model if a matching one is found.
        -l|--logging  Log output to log directory
"""
from docopt import docopt
print(docopt(__doc__, version='1.0.0rc2'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys

neccesary_args = ['<epochs>','<learning_rate>']
args = docopt(__doc__, version='1.0.0rc2')
keys = args.keys()

for k in neccesary_args:
    if k not in keys:
        print "MISSING KEY"
        exit(1)

if(args['<activation>'] not in ['RELU', 'SIG']):
    print "Nonvalid activation function"
    exit(1)

NUM_OF_EPOCHS = int(args['<epochs>'])
LEARNING_RATE = float(args['<learning_rate>'])
ACTIVATION = args['<activation>']
DROPOUT = float(args['<dropout>'])
MOMENTUM = float(args['<momentum>'])
WEIGHT_DECAY = float(args['<weight_decay>'])
DEPTH = int(args['<depth>'])

RUN_NAME = '_'.join(sys.argv[1:len(sys.argv)-1])
CONT_MODEL_PATH = 'models/' + RUN_NAME + '.model'
LOG_PATH = 'log/' + RUN_NAME + '.txt'
#TODO mkdir -p

cuda = torch.cuda.is_available()
#print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

#torch.manual_seed(42)
#if cuda:
#    torch.cuda.manual_seed(42)

batch_size = 100

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_dataset = datasets.CIFAR10('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                   ]))

num_train = len(train_dataset)
indices = list(range(num_train))
split = 40000

train_idx, valid_idx = indices[:split+1], indices[split+1:]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx) 

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)

validation_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=valid_sampler, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        
        return F.log_softmax(self.fc2(x), dim=1)

#ReLu stuff
class ReLuNet(nn.Module):
    def __init__(self):
        super(ReLuNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)

        return F.log_softmax(self.fc2(x), dim=1)
        
#Two Hidden Layer stuff
class TwoHidLayerNet(nn.Module):
    def __init__(self):
        super(TwoHidLayerNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 50)
        self.fc1_drop = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(DROPOUT)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.sigmoid(self.fc2(x))
        x = self.fc2_drop(x)

        return F.log_softmax(self.fc3(x), dim=1)

class TwoHidLayerReLuNet(nn.Module):
    def __init__(self):
        super(TwoHidLayerReLuNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 50)
        self.fc1_drop = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(DROPOUT)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)

        return F.log_softmax(self.fc3(x), dim=1)


if(args['--c'] or args['--cont']):
    print "Checking for model"
    #TODO test -e MODEL_PATH
        #load from file if exists, figure out what epoch and progress it was on last
    #TODO SIGINT trapping to save model
    #https://stackoverflow.com/questions/1112343/how-do-i-capture-sigint-in-python#1112350
    #TODO handling for resuming at the correct epoch
    #TODO logging to log folder

if(DEPTH == 1):
    if(ACTIVATION == 'SIG'):
        model = Net()
    elif(ACTIVATION == 'RELU'):
        model = ReLuNet()
    else:
        print "You shouldn't be here"
        exit(1)
else:
    if(ACTIVATION == 'SIG'):
        model = TwoHidLayerNet()
    elif(ACTIVATION == 'RELU'):
        model = TwoHidLayerReLuNet()
    else:
        print "You shouldn't be here"
        exit(1)

if cuda:
    model.cuda()
    
optimizer = optim.SGD(model.parameters(), weight_decay=WEIGHT_DECAY, lr=LEARNING_RATE, momentum=MOMENTUM)

#print(model)
def train(epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), split,
                100. * batch_idx / len(train_loader), loss))

def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        val_loss += F.nll_loss(output, target)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct / float(len(validation_loader.dataset) - split)
    accuracy_vector.append(accuracy)
    
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset) - split, accuracy))


def testidate(loss_vector, accuracy_vector):
    model.eval()
    test_loss, correct = 0, 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)
    loss_vector.append(test_loss)

    accuracy = 100. * correct / float(len(test_loader.dataset))
    accuracy_vector.append(accuracy)
    
    print('Testing set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))


epochs = NUM_OF_EPOCHS

val_lossv, val_accv = [], []
test_lossv, test_accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(val_lossv, val_accv)
testidate(test_lossv, test_accv)

print [float(x) for x in val_lossv]
print [float(x) for x in val_accv]

print [float(x) for x in test_lossv]
print [float(x) for x in test_accv]
