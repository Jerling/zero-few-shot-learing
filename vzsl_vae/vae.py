from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
#  from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
import sys
sys.path.append('./data')
from datasets import dataloader

H = 32
W = 32

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
    #  datasets.MNIST('../data', train=True, download=True,
                   #  transform=transforms.ToTensor()),
    #  batch_size=args.batch_size, shuffle=True, **kwargs)
#  test_loader = torch.utils.data.DataLoader(
    #  datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    #  batch_size=args.batch_size, shuffle=True, **kwargs)

data = dataloader()
keys = list(data.keys())
train_x = data[keys[3]] # (8855, 1024) 
train_att =data[keys[-1]] # (8855, 312) 
X_train, X_test, y_train, y_test = train_test_split(train_x, train_att, test_size=0.33, random_state=42)
train_loader = torch.utils.data.DataLoader(torch.cat((torch.from_numpy(X_train).float(),torch.from_numpy(y_train).float()),1), batch_size=args.batch_size, shuffle=True,**kwargs)
test_loader = torch.utils.data.DataLoader(torch.cat((torch.from_numpy(X_test).float(),torch.from_numpy(y_test).float()),1), batch_size=args.batch_size, shuffle=True,**kwargs)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc11 = nn.Linear(H*W, 800)
        self.fc12 = nn.Linear(312, 100)
        self.fc21 = nn.Linear(800, 312)
        self.fc22 = nn.Linear(800, 312)
        #  self.fc23 = nn.Linear(100, 312)
        #  self.fc24 = nn.Linear(100, 312)
        self.fc3 = nn.Linear(312, 800)
        self.fc4 = nn.Linear(800, H*W)

    def encode(self, x):
        h11 = F.relu(self.fc11(x))
        #  h12 = F.relu(self.fc12(a))
        return self.fc21(h11), self.fc22(h11)#, self.fc23(h12), self.fc24(h12)

    def reparameterize(self, mu, logvar):#, mu_a, sigma_a):
        #  print("mu_a", mu_a.shape)
        if self.training:
            #  std = torch.exp(0.5*logvar)
            #  eps = torch.randn_like(std)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            #  eps = torch.nn.init.normal(eps,mean=mu_a,std=sigma_a)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        #  print("x", x.shape)
        x_ = x[:,:1024]
        #  a = x[:,1024:]
        mu, logvar = self.encode(x_.view(-1, 1024))
        #  mu, logvar, mu_a, sigma_a = self.encode(x_.view(-1, 1024), a.view(-1, 312))
        z = self.reparameterize(mu, logvar)
        #  z = self.reparameterize(mu, logvar, mu_a, sigma_a)
        return self.decode(z), mu, logvar


#  W_mu = torch.ones(312, args.batch_size, requires_grad = True).to(device)
#  W_sigm = torch.ones(312, args.batch_size, requires_grad = True).to(device)


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1024), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD



def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #  print("data:",data.shape)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data[:,:1024], mu, logvar)
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
        for i, data in enumerate(test_loader):
            data = data[:,:1024].to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data.view(args.batch_size, 1, H, W)[:n],
                                      recon_batch.view(args.batch_size, 1, H,W)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, H, W),
                   'results/sample_' + str(epoch) + '.png')
