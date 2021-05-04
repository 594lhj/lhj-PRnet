import ofdm_util as util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from matplotlib import pyplot as plt
from PRnet import Autoencoder, Encoder, Decoder
import scipy.io as scio
from torchvision import datasets, transforms
from torch.autograd import Variable

S = 50000
N = 256
mu = 4
vS = 1000
snr = 20
BATCH_SIZE = 100
Epoch = 100

t = np.load('./data/train.npz')['data']
train = t.reshape(S * N, mu)

x = util.Mapping(train)
x = np.concatenate([np.expand_dims(x.real, 1), np.expand_dims(x.imag, 1)], axis=1).reshape(S, N, 2)
trainset = torch.tensor(x[vS:, :], dtype=torch.float)

torch_dataset = Data.TensorDataset(trainset, torch.zeros(trainset.shape))
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

valset = torch.tensor(x[:vS, :], dtype=torch.float)
val_dataset = Data.TensorDataset(valset, torch.zeros(valset.shape))
valloader = Data.DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, decode, d, out, alpha):
        Power1 = torch.sqrt(torch.sum(torch.square(out), dim=2))
        Max1 = torch.max(Power1, dim=1)[0]
        Mean1 = torch.mean(Power1, dim=1)
        Papr1 = torch.mean(Max1 / Mean1)
        # Max1 = torch.max(Power1)
        # Mean1 = torch.mean(Power1)
        # Papr1 = Max1 / Mean1
        return self.mse(decode, d) + alpha * Papr1


def train():
    encoder = Encoder(N)
    decoder = Decoder(N)
    autoencoder = Autoencoder(N, snr, encoder, decoder)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    myloss = My_loss()
    mse = nn.MSELoss()

    trainloss = []
    valloss = []

    for epoch in range(Epoch):
        epoch_cost = 0

        for index, (d, null) in enumerate(loader):
            decode, out = autoencoder(d)
            loss = myloss(decode, d, out, 0.1)
            epoch_cost += loss.data.numpy() / ((S - vS) / BATCH_SIZE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        vl = 0
        mses = 0
        for index, (v, null) in enumerate(valloader):
            decode, out = autoencoder(v)
            vloss = myloss(decode, v, out, 0.1)
            vl += vloss.data.numpy() / (vS / BATCH_SIZE)
            mseloss = mse(decode, v)
            mses += mseloss.data.numpy() / (vS / BATCH_SIZE)


        trainloss.append(epoch_cost)
        valloss.append(vl)
        print('epoch:', epoch + 1)
        print('trainloss:', epoch_cost)
        print('valloss:', vl)
        print('mseloss', mses)

    torch.save(autoencoder, './model/autoencoder.pkl')
    torch.save(encoder, './model/encoder.pkl')
    torch.save(decoder, './model/decoder.pkl')

    plt.plot(trainloss, 'b', label="train")
    plt.plot(valloss, 'r', label="val")
    plt.show()


if __name__ == '__main__':
    train()
