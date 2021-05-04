import ofdm_util as util
import numpy as np
import torch
from PRnet import Autoencoder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from matplotlib import pyplot as plt
import scipy.io as scio
from torchvision import datasets, transforms
from torch.autograd import Variable


S = 1000
N = 256
mu = 4
snr = 20
BATCH_SIZE = 100


t = np.load('./data/test.npz')['data']
test = t.reshape(S * N, mu)

x = util.Mapping(test)
x = np.concatenate([np.expand_dims(x.real, 1), np.expand_dims(x.imag, 1)], axis=1).reshape(S, N, 2)
testset = torch.tensor(x, dtype=torch.float)


tx=torch.zeros((S,N,2))
ty=torch.zeros((S,N,2))
encoder = torch.load('./model/encoder.pkl')
for k in range(S//BATCH_SIZE):
    d=testset[k*BATCH_SIZE:k*BATCH_SIZE+BATCH_SIZE]
    out = encoder(d)
    tx[k*BATCH_SIZE:k*BATCH_SIZE+BATCH_SIZE]=out
    ty[k*BATCH_SIZE:k*BATCH_SIZE+BATCH_SIZE]=d

tx=tx.detach().numpy()
ty=ty.detach().numpy()
prnet=(tx[:,:,0]+1j*tx[:,:,1])
origin=(ty[:,:,0]+1j*ty[:,:,1])
print(prnet)
print(origin)

scio.savemat('./mat/prnet.mat', {'prnet': prnet})
scio.savemat('./mat/origin.mat', {'origin': origin})


