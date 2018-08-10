import sys
sys.path.append(".")
from data.datasets import dataloader
import numpy as np
import torch as t
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

data = dataloader("CUB1")
print(data.keys())

features = data['features'].T
labels = data['labels']
train_loc = data['train_loc']
x_train = t.Tensor([features[i] for i in np.reshape(train_loc,(train_loc.shape[0]))])
y_train = t.Tensor([labels[i] for i in np.reshape(train_loc,(train_loc.shape[0]))])

trainloader = DataLoader(t.cat((x_train, y_train),1),32,True)

print(x_train.shape)
print(y_train.shape)

class NLP(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super(NLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)
        #  self.sfmx = nn.Softmax(out_features);

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

lr = 1e-2
mtm = 9e-1
epochs = 10
display = 50
nlp = NLP(x_train.shape[1], 512, 200)

print(nlp)

criterion = nn.MSELoss(size_average=None, reduce=None, reduction='elementwise_mean')
optimizer = optim.Adam(nlp.parameters(), lr=lr)

nlp.training = True
for epoch in range(epochs):
    running_loss = .0
    for i, data in enumerate(trainloader,0):
        print(data.shape)
        inputs = Variable(data[:,:-1])
        labels = Variable(data[:,-1])
        print(type(labels))
        optimizer.zero_grad()
        outputs = t.argmax(nlp(inputs), dim=1)
        print(outputs.shape)
        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if i % 50 == display - 1:
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss/display))
       
print("Finised Training")
