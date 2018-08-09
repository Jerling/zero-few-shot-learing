from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
#  from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import sys
sys.path.append('./data')
from datasets import dataloader

H = 32
W = 32

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#  train_loader = torch.utils.data.DataLoader(
    #  datasets.MNIST('/tmp/data', train=True, download=False,
                   #  transform=transforms.ToTensor()),
    #  batch_size=args.batch_size, shuffle=True, **kwargs)
#  test_loader = torch.utils.data.DataLoader(
    #  datasets.MNIST('/tmp/data', train=False, transform=transforms.ToTensor()),
    #  batch_size=args.batch_size, shuffle=True, **kwargs)
data = dataloader()
keys = list(data.keys())
train_x = data[keys[3]] # (8855, 1024) 
train_att =data[keys[-1]] # (8855, 312) 
train_loader = torch.utils.data.DataLoader(torch.from_numpy(np.concatenate((train_x,train_att), axis=1)).float(),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(torch.from_numpy(np.concatenate((train_x,train_att), axis=1)).float(),
    batch_size=args.batch_size, shuffle=True, **kwargs)



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(H*W, 800)
        self.fc21 = nn.Linear(800, 400)
        self.fc22 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 800)
        self.fc4 = nn.Linear(800, H*W)

        self.fca = nn.Linear(312, 600, bias=True)
        self.fca1 = nn.Linear(600, 400, bias=True)
        self.fca2 = nn.Linear(600, 400, bias=True)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def sample(self, a):
        ha = F.relu(self.fca(a))
        return self.fca1(ha), self.fca2(ha)

    def reparameterize(self, mu, logvar, amu, alogvar):
        #  print("mu:",mu.shape)
        #  print("lv:",logvar.shape)
        if self.training:
            alogvar = torch.var(alogvar,dim=0)
            amu = torch.mean(amu,dim=0)
            #  std = torch.diag(torch.exp(alogvar))
            #  print(alogvar.size())
            #  print(std.size())
            #  print(amu.size())
            #  print("std:",std.shape)
            eps = torch.normal(mean=amu,std=alogvar)
            return eps.mul(logvar).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x, a):
        #  print("x:",x.shape)
        mu, logvar = self.encode(x.view(-1, H*W))
        amu, alogvar = self.sample(a)
        z = self.reparameterize(mu, logvar, amu, alogvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #  BCE = F.binary_cross_entropy(recon_x, x.view(-1, H*W), size_average=False)
    m = torch.nn.Sigmoid()
    loss = torch.nn.BCELoss(size_average=False)
    BCE = loss(m(recon_x), x)
    #  print("BCE:", BCE.data)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #  print("KLD:", KLD.data)

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data_att in enumerate(train_loader):
        data = data_att[:,:1024]
        att = data_att[:,1024:]
        data = data.to(device)
        #  print(data.size())
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, att)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data_att in enumerate(test_loader):
            data = data_att[:,:1024]
            att = data_att[:,1024:]
            data = data.to(device)
            recon_batch, mu, logvar = model(data, att)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0 and epoch % 10 == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data.view(args.batch_size, 1, H, W)[:n],
                                      recon_batch.view(args.batch_size, 1, H, W)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    if epoch % 10 == 0:
        with torch.no_grad():
            sample = torch.randn(64, 400).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, H, W),
                       'results/sample_' + str(epoch) + '.png')
