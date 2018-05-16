"""Usage:
	opgg_leaderboard_scraper.py <activation> <epochs> <learning_rate>

Arguments:
    activation: SIG - Sigmoid Function
        RELU - ReLu Activation Function
	epochs: INTEGER
	learning_rate: FLOAT
"""
from docopt import docopt
print(docopt(__doc__, version='1.0.0rc2'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

cuda = torch.cuda.is_available()
#print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

#torch.manual_seed(42)
#if cuda:
#    torch.cuda.manual_seed(42)

batch_size = 20

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

validation_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 100)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.sigmoid(self.fc2(x))
        x = self.fc2_drop(x)
        
        return F.log_softmax(self.fc3(x), dim=1)
#ReLu stuff
class ReLuNet(nn.Module):
    def __init__(self):
        super(ReLuNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 100)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)

        return F.log_softmax(self.fc3(x), dim=1)
        
if(ACTIVATION == 'SIG'):
    model = Net()
elif(ACTIVATION == 'RELU'):
    model = ReLuNet()
else:
    print "You shouldn't be here"
    exit(1)

if cuda:
    model.cuda()
    
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.5)

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
                epoch, batch_idx * len(data), len(train_loader.dataset),
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

    accuracy = 100. * correct / float(len(validation_loader.dataset))
    accuracy_vector.append(accuracy)
    
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))

epochs = NUM_OF_EPOCHS

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    if(epoch < 4):
        train(epoch)
    validate(lossv, accv)

print "\n"
print zip(range(1, epochs + 1), zip([float(x) for x in lossv] , [float(x) for x in accv]))


