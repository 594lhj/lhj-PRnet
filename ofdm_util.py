import numpy as np
import torch

mapping_table = {
    (0, 0, 0, 0): 1 + 1j,
    (0, 0, 0, 1): -1 + 1j,
    (0, 0, 1, 0): 1 - 1j,
    (0, 0, 1, 1): -1 - 1j,
    (0, 1, 0, 0): 3 + 1j,
    (0, 1, 0, 1): -3 + 1j,
    (0, 1, 1, 0): 3 - 1j,
    (0, 1, 1, 1): -3 - 1j,
    (1, 0, 0, 0): 1 + 3j,
    (1, 0, 0, 1): -1 + 3j,
    (1, 0, 1, 0): 1 - 3j,
    (1, 0, 1, 1): -1 - 3j,
    (1, 1, 0, 0): 3 + 3j,
    (1, 1, 0, 1): -3 + 3j,
    (1, 1, 1, 0): 3 - 3j,
    (1, 1, 1, 1): -3 - 3j
}

demapping_table0 = {v: k for k, v in mapping_table.items()}


def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])


def map0(bit):
    return mapping_table[tuple(bit)]


def IDFT(ofdm_data):
    return torch.ifft(ofdm_data)


def channel(x, snr, N):
    snr = 10 ** (snr / 10.0)
    Power = torch.sum(torch.sum(torch.square(x), dim=2), dim=1) / N
    npower = torch.sqrt((Power / snr) / 2)
    aa = torch.randn((len(x), len(x[0]), len(x[0, 0, :])))
    bb = npower.unsqueeze(1).unsqueeze(1).repeat(1, 256, 2)
    return aa.mul(bb)
