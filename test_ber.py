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
BATCH_SIZE = 100


t = np.load('./data/test.npz')['data']
test = t.reshape(S * N, mu)

x = util.Mapping(test)
x = np.concatenate([np.expand_dims(x.real, 1), np.expand_dims(x.imag, 1)], axis=1).reshape(S, N, 2)
testset = torch.tensor(x, dtype=torch.float)

encoder = torch.load('./model/encoder.pkl')
decoder = torch.load('./model/decoder.pkl')

snr_list = np.arange(0, 32, 1)
ber_list = np.zeros(len(snr_list))

bit_send=t
# for i, snr in enumerate(snr_list):
#     re = torch.zeros((S, N, 2))
#     for k in range(S // BATCH_SIZE):
#         d = testset[k * BATCH_SIZE:k * BATCH_SIZE + BATCH_SIZE]
#
#         encode = encoder(d)
#         out = torch.ifft(encode, 1)
#         x = util.channel(out, snr, N) + out
#         y = torch.fft(x, 1)
#         decode = decoder(y)
#
#         re[k * BATCH_SIZE:k * BATCH_SIZE + BATCH_SIZE] = decode
#
#     re = re.detach().numpy()
#     bit_recv=util.channales(re)
#
#     ber_list[i] = np.sum(np.logical_xor(bit_send, bit_recv)) / bit_send.size
#     print('%d' % i)

for i, snr in enumerate(snr_list):
    re = torch.zeros((S, N, 2))
    for k in range(S):
        d = testset[k,:]
        out = torch.ifft(d,1)

        snr = 10 ** (snr / 10.0)
        xpower = torch.sum(torch.sum(torch.square(out), dim=1),dim=0)/len(d)
        npower = (xpower / snr)/2
        noise = np.random.randn(len(out)) * np.sqrt(npower)
        x=out+noise

        y = np.fft.fft(x,0)
        re[k] = y

    bit_recv=util.channales(re)
    ber_list[i] = np.sum(np.logical_xor(bit_send, bit_recv)) / bit_send.size
    print('%d' % i)

plt.plot(snr_list, ber_list, label='prnet', color='b', marker='o')
plt.show()

