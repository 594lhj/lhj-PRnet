import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import torch


k = 64  # numbel of ofdm subcarriers
cp = k / 4  # numbel of pilot carriers per ofdm blcok
pilotvalue = 3 + 3j
print(k)
print('python')
p = 8  # numbel of pilot carrier per ofdm block
allcarriers = np.arange(k)

pilotcarriers = allcarriers[::k // p]
pilotcarriers = np.hstack([pilotcarriers, np.array([allcarriers[-1]])])

# data carrier are all ramaining carriers
datacarrier = np.delete(allcarriers, pilotcarriers)
p = p + 1
print("pilotcarrier :%s" % pilotcarriers)
print("datacarrier :%s" % datacarrier)
print("allcarrier :%s" % allcarriers)

plt.show()
plt.plot(pilotcarriers, np.zeros_like(pilotcarriers), 'bo', label='pilot')
plt.plot(datacarrier, np.zeros_like(datacarrier), 'ro', label='data')
plt.show()
plt.legend()
mu = 4  # bits per symbol
payloadbit_per_ofdm = len(datacarrier) * 4  # numbel of payload bits per ofdm symbol
mapping_table = {
    (0, 0, 0, 0): -3 - 3j,
    (0, 0, 0, 1): -3 - 1j,
    (0, 0, 1, 0): -3 + 3j,
    (0, 0, 1, 1): -3 + 1j,
    (0, 1, 0, 0): -1 - 3j,
    (0, 1, 0, 1): -1 - 1j,
    (0, 1, 1, 0): -1 + 3j,
    (0, 1, 1, 1): -1 + 1j,
    (1, 0, 0, 0): 3 - 3j,
    (1, 0, 0, 1): 3 - 1j,
    (1, 0, 1, 0): 3 + 3j,
    (1, 0, 1, 1): 3 + 1j,
    (1, 1, 0, 0): 1 - 3j,
    (1, 1, 0, 1): 1 - 1j,
    (1, 1, 1, 0): 1 + 3j,
    (1, 1, 1, 1): 1 + 1j
}

for b3 in [0, 1]:
    for b2 in [0, 1]:
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                B = (b3, b2, b1, b0)
                Q = mapping_table[B]
                plt.plot(Q.real, Q.imag, 'bo')
plt.show()
plt.legend()
demapping_table = {v: k for k, v in mapping_table.items()}

channelresponse = np.array([1, 0, 0.3 + 0.3j])
H_exact = np.fft.fft(channelresponse, k)
plt.plot(allcarriers, abs(H_exact))
plt.show()
plt.legend()

SNRdb = 100

bits = np.random.binomial(n=1, p=0.5, size=(payloadbit_per_ofdm,))
print("Bits count: ", len(bits))
print("First 20 bits: ", bits[:20])
print("Mean of bits (should be around 0.5): ", np.mean(bits))


def SP(bits):
    return bits.reshape((len(datacarrier), mu))


bits_sp = SP(bits)
print("the first 5 bit group")
print(bits_sp[:5, :])


def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])


QAM = Mapping(bits_sp)
print("the first 5 qam symbols and bits")
print(bits_sp[:5, :])
print(QAM[:5])
print(len(QAM))


def ofdm_symbol(Qam_payload):
    symbol = np.zeros(k, dtype=complex)
    symbol[pilotcarriers] = pilotvalue
    symbol[datacarrier] = Qam_payload
    return symbol


ofdm_data = ofdm_symbol(QAM)
print("numbel of ofdm  carriers in frequency domain:", len(ofdm_data))


def IDFT(ofdm_data):
    return np.fft.ifft(ofdm_data)


ofdm_time = IDFT(ofdm_data)
print("number of  ofdm samples in time-domain before cp:", len(ofdm_time))


def addcp(ofdm_time):
    CP_data = ofdm_time[int(-cp):]
    return np.hstack([CP_data, ofdm_time])


ofdm_withcp = addcp(ofdm_time)
print("number of ofdm symbel in time-domain with cp :", len(ofdm_withcp))


def channel(signal):
    convolved = np.convolve(signal, channelresponse)
    signal_power = np.mean(abs(convolved))
    sigma2 = signal_power * 10 ** (-SNRdb / 10)  # calcuate noise power based on signal power and snr
    print("Rx signal powers is %.4f,Noise power :%.4f" % (signal_power, sigma2))
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    print("convolved len is %s :", len(convolved))
    return convolved + noise


ofdm_tx = ofdm_withcp
ofdm_rx = channel(ofdm_tx)
plt.plot(abs(ofdm_tx), 'b', label="tx signal")
plt.plot(abs(ofdm_rx), 'r', label="rx signal")
plt.legend()

plt.xlabel("time")
plt.ylabel('$|x(t)|$')
plt.show()


def removeCP(signal):
    return signal[int(cp):int(cp) + k]


ofdm_rx_nocp = removeCP(ofdm_rx)


def DFT(signal):
    return np.fft.fft(signal)


ofdm_demod = DFT(ofdm_rx_nocp)


def channelEstimate(ofdm_demod):
    pilots = ofdm_demod[pilotcarriers]
    hest_at_pilots = pilots / pilotvalue
    hest_abs = interp1d(pilotcarriers, abs(hest_at_pilots), kind='linear')(allcarriers)
    hest_phase = interp1d(pilotcarriers, np.angle(hest_at_pilots), kind='linear')(allcarriers)
    hest = hest_abs * np.exp(1j * hest_phase)
    plt.plot(allcarriers, abs(H_exact), label='correct channel')
    plt.stem(pilotcarriers, abs(hest_at_pilots), label='pilot estimates')
    plt.plot(allcarriers, abs(hest_abs), label='estimate channel via interplation')
    plt.grid(True)
    plt.xlabel('carrier index')
    plt.ylabel('$|H(f)|$')
    plt.legend()
    plt.ylim(0, 2)
    plt.show()
    return hest


Hest = channelEstimate(ofdm_demod)


def equlize(ofdm_demod, hest):
    return ofdm_demod / hest


equlized_hest = equlize(ofdm_demod, Hest)


def get_payload(equlized):
    return equlized[datacarrier]


QAM_test = get_payload(equlized_hest)
plt.plot(QAM_test.real, QAM_test.imag, 'bo')
plt.show()


def Demapping(QAM):
    constellation = np.array([x for x in demapping_table.keys()])
    dists = abs(QAM.reshape((-1, 1)) - constellation.reshape((1, -1)))
    index = dists.argmin(axis=1)
    hardDecision = constellation[index]
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision


PS_est, hardDecision = Demapping(QAM_test)

for qam, hard in zip(QAM_test, hardDecision):
    plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
    plt.plot(hard.real, hard.imag, 'ro')
plt.show()


def P2S(ps_est):
    return ps_est.reshape((1, -1))


bits_est = P2S(PS_est)

error_rate = np.sum(abs(bits - bits_est)) / len(bits)

print("obtained bit error rate is :", error_rate)