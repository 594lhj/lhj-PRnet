clear;
clc;

load('prnet.mat')
load('origin.mat')

N=256;  %子载波数
L=1;    %过采样倍数
sim_prd=1000;   %OFDM符号数

prnet_data = ifft(prnet,[],2)*sqrt(N);
signal_power = abs((prnet_data).^2);
mean_power = mean(signal_power,2);
peak_power = max(signal_power,[],2);
papr = 10*log10(peak_power./mean_power);
[cdf_prnet,papr_prnet] = ecdf(papr);

origin_data = ifft(origin,[],2)*sqrt(N);
signal_power = abs((origin_data).^2);
mean_power = mean(signal_power,2);
peak_power = max(signal_power,[],2);
papr = 10*log10(peak_power./mean_power);
[cdf_origin,papr_origin] = ecdf(papr);

figure(1);
semilogy(papr_prnet,1-cdf_prnet,'-o',papr_origin,1-cdf_origin,'-o');
legend('prnet','origin')
xlabel('PAPR_0(dB)')
ylabel('CCDF (PAPR > PAPR0)')
grid on