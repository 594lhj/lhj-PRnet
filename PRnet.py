import ofdm_util as util
import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, N, snr, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.N = N
        self.snr = snr
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encode = self.encoder(x)
        out = torch.ifft(encode, 1)
        x = util.channel(out, self.snr, self.N) + out
        # x = out
        y = torch.fft(x, 1)
        decode = self.decoder(y.view(-1, self.N * 2)).reshape(-1, self.N, 2)
        return decode, out


class Encoder(nn.Module):
    def __init__(self, N):
        super(Encoder, self).__init__()
        self.N=N
        self.encoder = nn.Sequential(
            nn.Linear(N * 2, N * 2),
            nn.BatchNorm1d(N * 2),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(N * 2, N * 2),
            nn.BatchNorm1d(N * 2),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            # nn.Linear(N * 2, N * 2),
            # nn.BatchNorm1d(N * 2),
            # nn.Dropout(0.5),
            # nn.ReLU(inplace=True),

            nn.Linear(N * 2, N * 2))

    def forward(self, x):
        return self.encoder(x.view(-1, self.N * 2)).reshape(-1, self.N, 2)


class Decoder(nn.Module):
    def __init__(self, N):
        super(Decoder, self).__init__()
        self.N=N
        self.decoder = nn.Sequential(
            nn.Linear(N * 2, N * 2),
            nn.BatchNorm1d(N * 2),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(N * 2, N * 2),
            nn.BatchNorm1d(N * 2),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            # nn.Linear(N * 2, N * 2),
            # nn.BatchNorm1d(N * 2),
            # nn.Dropout(0.5),
            # nn.ReLU(inplace=True),

            nn.Linear(N * 2, N * 2))

    def forward(self, x):
        return self.decoder(x.view(-1, self.N * 2)).reshape(-1, self.N, 2)

