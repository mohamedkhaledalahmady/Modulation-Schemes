%% Clear Section
clear
close all
clc

%% BPSK
%% Generate data bit stream
number_of_bits = 1000000;                   % number of transmitted bits
threshold = 0;                              % threshold value
bits = randi([0 1], 1, number_of_bits);     % generate random bits
Tx_Data = 2*bits-1;                         % mapping to BPSK
%% Add AWGN from Channel
Eb = 1;                                                 % energy of bit
SNR = -2:0.5:20;                                        % SNR valus
No = Eb./(10.^(SNR/10));                                % variance value of noise
noise = (sqrt(No/2))'.*randn(1, number_of_bits);        % generate noise with zero mean and No/2 variance
%% Add Noise to Tx data
Rx_Data = Tx_Data + noise;                              % adding noise to Tx data
%% BER Calculations
BER_Calculated_BPSK = zeros(1, length(SNR));
BER_Theoretical_BPSK = 0.5*erfc(sqrt(Eb./No));
for i = 1:length(SNR)
    BER_Calculated_BPSK(1, i) = (sum(sign(Rx_Data(i, :)) ~= Tx_Data))/number_of_bits;
end
figure
semilogy(SNR, BER_Calculated_BPSK, 'LineWidth', 2)
hold on
semilogy(SNR, BER_Theoretical_BPSK, 'LineWidth', 2)
xlim([-2 12])
xlabel("Eb/No (dB)")
ylabel("Bit Error Rate (BER)")
title("BER vs. SNR Curve for BPSK")
legend("Calculated BER", "Theoretical BER")
grid on
hold off

%% QPSK_1
%% Generate data bit stream
number_of_bits = 1000000;                       % number of transmitted bits
M = 4;                                          % number of symbols
bits = randi([0 1], 1, number_of_bits);         % generate random bits
Tx_Data = mapping_QPSK_1(bits);                 % determine Tx data
%% Add AWGN from Channel
Eb = 1;                                         % energy of bit
SNR = -2:0.5:20;                                % SNR values
No = Eb./(10.^(SNR/10));                        % variance value of noise
noise = (sqrt(No/2))'.*randn(1, number_of_bits/2) + 1j*(sqrt(No/2))'.*randn(1, number_of_bits/2); % generate noise with zero mean and No/2 variance
%% Add Noise to Tx data
Rx_Data = Tx_Data + noise;                      % adding noise to Tx data
%% BER Calculations
BER_Calculated_QPSK_1 = zeros(1, length(SNR));
BER_Theoretical_QPSK = 0.5 * erfc(sqrt(Eb./No));
R = real(Rx_Data);
I = imag(Rx_Data);
extract_Data(1:size(R, 1), 1:2:2 * size(R, 2)) = R;
extract_Data(1:size(I, 1), 2:2:2 * size(I, 2)) = I;
Data = 2*bits-1;
for i = 1:length(SNR)
    BER_Calculated_QPSK_1(1, i) = (sum(sign(extract_Data(i, :)) ~= Data)) / number_of_bits;
end
figure
semilogy(SNR, BER_Calculated_QPSK_1, 'LineWidth', 2)
xlim([-2 12])
hold on
semilogy(SNR, BER_Theoretical_QPSK, 'LineWidth', 2)
xlabel("Eb/No (dB)")
ylabel("Bit Error Rate (BER)")
title("BER vs. SNR Curve for QPSK")
legend("Calculated BER", "Theoretical BER")
grid on
hold off

%% QPSK_2
%% Generate data bit stream
Data = randi([0 1], 1, number_of_bits);             % generate random bits
Tx_Data = mapping_QPSK_2(Data);                     % determine Tx data
%% Add Noise to Tx data
Rx_Data = Tx_Data + noise;                          % adding noise to Tx data
%% BER Calculations
BER_Calculated_QPSK_2 = zeros(1, length(SNR));
for i = 1:length(SNR)
    BER_Calculated_QPSK_2(1, i) = BER_QPSK_2(Data, Rx_Data(i, :))/number_of_bits;
end
figure
semilogy(SNR, BER_Calculated_QPSK_1, 'LineWidth', 2)
xlim([-2 12])
hold on
semilogy(SNR, BER_Calculated_QPSK_2, 'LineWidth', 2)
xlabel("Eb/No (dB)")
ylabel("Bit Error Rate (BER)")
title("BER vs. SNR Curve for QPSK")
legend("Calculated QPSK_1 BER", "Calculated QPSK_2 BER")
grid on
hold off

%% 8PSK
%% Generate data bit stream
number_of_bits = 1000000 - 1;                         % number of transmitted bits
M = 8;                                              % number of symbols
bits = randi([0 1], 1, number_of_bits);             % generate random bits
Tx_Data = mapping_8PSK(bits);                       % determine Tx data
%% Add AWGN from Channel
bits = log2(M);
Eb = 1;
E = Eb * bits;
SNR = -2:0.5:20;                                    % SNR values
No = Eb ./ (10 .^ (SNR / 10));                      % variance value of noise
noise = (sqrt(No / 2))' .* randn(1, number_of_bits / bits) + 1j * (sqrt(No / 2))' .* randn(1, number_of_bits / bits); % generate noise with zero mean and No/2 variance
%% Add Noise to Tx data
Rx_Data = Tx_Data + noise; % adding noise to Tx data
%% BER Calculations
SER_Theoretical_8PSK = erfc(sqrt(E./No)*sin(pi/M));
BER_Exact_8PSK = berawgn(SNR, 'psk', M, 'nondiff');
SER_Calculated_8PSK = zeros(1, length(SNR));
for i = 1:length(SNR)
    SER_Calculated_8PSK(1, i) = angle_8PSK(Tx_Data, Rx_Data(i, :))/(number_of_bits/log2(M));
end
BER_Calculated_8PSK = SER_Calculated_8PSK/bits;
BER_Theoretical_8PSK = SER_Theoretical_8PSK/bits;
figure
semilogy(SNR, BER_Calculated_8PSK, 'LineWidth', 2)
hold on
semilogy(SNR, BER_Theoretical_8PSK, 'LineWidth', 2)
% hold on
% semilogy(SNR, BER_Exact_8PSK, 'LineWidth', 2)
xlim([-2 12])
xlabel("Eb/No (dB)")
ylabel("Error Rate")
title("BER vs. SNR Curve for 8PSK")
legend("Calculated BER", "Theoretical BER")% "Exact BER")
grid on
hold off

%% 16QAM
%% Generate data bit stream
number_of_bits = 1000000;                         % number of transmitted bits
M = 16;                                         % number of symbols
data_bits = randi([0 1], 1, number_of_bits);    % generate random bits
Tx_Data = mapping_16QAM(data_bits);             % determine Tx data
%% Add AWGN from Channel
bits = log2(M);
Eb = 2.5;
E = Eb * 0.4;
SNR = -2:0.5:20;                                % SNR values
No = Eb ./ (10 .^ (SNR / 10));                  % variance value of noise
noise = (sqrt(No/2))'.*randn(1, number_of_bits/bits)+ 1j * (sqrt(No / 2))' .* randn(1, number_of_bits / bits); % generate noise with zero mean and No/2 variance
%% Add Noise to Tx data
Rx_Data = Tx_Data + noise;                      % adding noise to Tx data
%% BER Calculations
SER_Theoretical_16QAM = 1.5 * erfc(sqrt(E./No));
BER_Calculated_16QAM = zeros(1, length(SNR));
for i = 1:length(SNR)
    BER_Calculated_16QAM(1, i) = BER_16QAM(data_bits, Rx_Data(i, :))/(number_of_bits);
end
BER_Theoretical_16QAM = SER_Theoretical_16QAM/bits;
figure
semilogy(SNR, BER_Calculated_16QAM, 'LineWidth', 2)
hold on
semilogy(SNR, BER_Theoretical_16QAM, 'LineWidth', 2)
xlim([-2 12])
xlabel("Eb/No (dB)")
ylabel("Error Rate")
title("BER vs. SNR Curve for 16QAM")
legend("Calculated BER", "Theoretical BER")
grid on
hold off

%% FSK
%% Generate data bit stream
number_of_bits = 1000000;                       % number of transmitted bits
amplitude = 1;                                  % amplitude of transmitted signal
data_bits = randi([0 1], 1, number_of_bits);    % generate random bits
Tx_Data = data_bits;                            % determine Tx data
Tx_Data(Tx_Data == 1) = 1j;
Tx_Data(Tx_Data == 0) = 1;
%% Add AWGN from Channel
Eb = 1;
SNR = -2:0.5:20; % SNR values
No = Eb ./ (10 .^ (SNR / 10));                  % variance value of noise
noise = (sqrt(No / 2))' .* randn(1, number_of_bits) + 1j * (sqrt(No / 2))' .* randn(1, number_of_bits); % generate noise with zero mean and No/2 variance
%% Add Noise to Tx data
Rx_Data = Tx_Data + noise;                      % adding noise to Tx data
%% BER Calculations
BER_Theoretical_FSK = 0.5*erfc(sqrt(E./(2*No)));
BER_Calculated_FSK = zeros(1, length(SNR));
for i = 1:length(SNR)
    BER_Calculated_FSK(1, i) = BER_FSK(data_bits, Rx_Data(i, :))/(number_of_bits);
end
figure
semilogy(SNR, BER_Calculated_FSK, 'LineWidth', 2)
hold on
semilogy(SNR, BER_Theoretical_FSK, 'LineWidth', 2)
xlim([-2 12])
xlabel("Eb/No (dB)")
ylabel("Error Rate")
title("BER vs. SNR Curve for 16QAM")
legend("Calculated BER", "Theoretical BER")
grid on
hold off

%% Comparison
figure
semilogy(SNR, BER_Calculated_BPSK, 'LineWidth', 2)
hold on
semilogy(SNR, BER_Theoretical_BPSK, '*', 'LineWidth', 1)
hold on
semilogy(SNR, BER_Calculated_QPSK_1, 'LineWidth', 2)
hold on
semilogy(SNR, BER_Theoretical_QPSK, '+', 'LineWidth', 1)
hold on
semilogy(SNR, BER_Calculated_QPSK_2, 'LineWidth', 2)
hold on
semilogy(SNR, BER_Calculated_8PSK, 'LineWidth', 2)
hold on
semilogy(SNR, BER_Theoretical_8PSK, 'o', 'LineWidth', 1)
hold on
semilogy(SNR, BER_Calculated_16QAM, 'LineWidth', 2)
hold on
semilogy(SNR, BER_Theoretical_16QAM, '*', 'LineWidth', 1)
hold on
semilogy(SNR, BER_Calculated_FSK, 'LineWidth', 2)
hold on
semilogy(SNR, BER_Theoretical_FSK, '+', 'LineWidth', 1)
xlabel("Eb/No (dB)")
ylabel("Bit Error Rate")
title("BER vs. SNR Curve")
ylim([10^-5 10^0])
legend("Calculated BPSK", "Theoretical BPSK", "Calculated QPSK_1", "Theoretical QPSK", ...
    "Calculated QPSK_2", "Calculated 8PSK", "Theoretical 8PSK", "Calculated 16QAM", "Theoretical 16QAM", "Calculated FSK", "Theoretical FSK")
grid on
hold off

%% PSD for FSK
%% Generate Ensample
clear
clc
close all
number_of_enamples=500;                          % total number of Realizations
number_of_bits=100;                              % number of bits in each Realization
Tb=7;                                            % number of samples for each bit
number_of_samples=number_of_bits*Tb;             % total number of samples for each Realization
Total_Transmitted_Data=zeros(number_of_enamples, number_of_samples-1);  % hold transmitted data
for i = 1 : number_of_enamples
    if i == 1
        Total_Transmitted_Data = Generate_Random_Data();
    else
        Total_Transmitted_Data = [Total_Transmitted_Data; Generate_Random_Data()];
    end
end
%%Calculate Statistical AutoCorrelation function (ACF)
average=zeros(1, number_of_samples);
for i = 1 : number_of_samples
    average(1,i) = mean((Total_Transmitted_Data(:,1)).*conj((Total_Transmitted_Data(:,i))));
end
figure
plot(average)
grid on
xlabel("\tau")
ylabel("ACF")
title("Statistical ACF")
%%Calculate PSD
PSD = fftshift(fft(average));
N = length(PSD);                                % Number of samples
Ts=1;                                           % sample duration every 1 sec
Fs=1/Ts;                                        % sampling frequency
freq =(-N/2+1:N/2)*(Fs/(N));                    % Frequency axis (Hz)
figure
plot(freq, abs(PSD)/N, 'linewidth', 2);                 
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('PSD of BaseBand Signal');
grid on

%%shift PSD
fc=1e1;
freq1 = ((-N/2+1:N/2)*(Fs/(N))+fc);             % Frequency axis (Hz)
freq2 = ((-N/2+1:N/2)*(Fs/(N))-fc);             % Frequency axis (Hz)
figure
plot(freq1, abs(PSD)/N, freq2, abs(PSD)/N, 'linewidth', 2);                 
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('PSD of PassBand Signal');
grid on

%% Useful Functions
%% Mapping QPSK_1
function y = mapping_QPSK_1(x)
    M = 4;
    bits = log2(M);
    Eb = 1;
    E = bits * Eb;
    split_x = reshape(x, bits, length(x) / bits)';
    split_x_indices = bi2de(split_x, 'left-msb')+1;
    y = zeros(length(split_x_indices), 1);
    phases = [5*pi/4, 3*pi/4, 7*pi/4, pi/4];
    y = sqrt(E)*exp(1j*phases(split_x_indices));
end
%% Mapping QPSK_2
function y = mapping_QPSK_2(x)
    M = 4;
    bits = log2(M);
    Eb = 1;
    E = bits*Eb;
    split_x = reshape(x, bits, length(x) / bits)';
    split_x_indices = bi2de(split_x, 'left-msb')+1;
    y = zeros(length(split_x_indices), 1);
    phases = [5*pi/4, 3*pi/4, pi/4, 7*pi/4];
    y = sqrt(E)*exp(1j*phases(split_x_indices));
end
%% BER QPSK_2
function incorrect_bits = BER_QPSK_2(Tx_bits, Rx)
    M = 4;
    bits = log2(M);
    Rx_bits = zeros(1, bits * length(Rx));
    k = 1;
    R = real(Rx);
    I = imag(Rx);
    Rx_bits(1:2:2*length(R)) = (sign(R)>=0);         % b0
    Rx_bits(2:2:2*length(I)) = (sign(R) ~= sign(I)); % b1
    incorrect_bits = sum(Rx_bits ~= Tx_bits);
end
%% Mapping to 8PSK
function y = mapping_8PSK(x)
    M = 8;
    bits = log2(M);
    Eb = 1;
    E = bits * Eb;
    split_x = reshape(x, bits, length(x)/bits)';
    split_x_indices = bi2de(split_x, 'left-msb')+1;
    y = zeros(1, length(split_x));
    phases = [0, pi/4, 3*pi/4, pi/2, 7*pi/4, 3*pi/2, pi, 5*pi/4];
    y = sqrt(E)*exp(1j*phases(split_x_indices));
end
%% Determine Angle of sympol in 8PSK number
function incorrecr_bits = angle_8PSK(Tx, Rx)
    Tx_angles = rad2deg(mod(angle(Tx), 2 * pi));
    Rx_angles = rad2deg(mod(angle(Rx), 2 * pi));
    Tx_sympols = zeros(1, length(Tx));
    Rx_sympols = zeros(1, length(Rx));
    for i = 1:length(Tx_angles)
        if (Tx_angles(i) >= 0 && Tx_angles(i) < 22.5) || (Tx_angles(i) >= 337.5 && Tx_angles(i) < 360)
            Tx_sympols(i) = 1;
        end
        if (Tx_angles(i) >= 22.5 && Tx_angles(i) < 67.5)
            Tx_sympols(i) = 2;
        end
        if (Tx_angles(i) >= 67.5 && Tx_angles(i) < 112.5)
            Tx_sympols(i) = 3;
        end
        if (Tx_angles(i) >= 112.5 && Tx_angles(i) < 157.5)
            Tx_sympols(i) = 4;
        end
        if (Tx_angles(i) >= 157.5 && Tx_angles(i) < 202.5)
            Tx_sympols(i) = 5;
        end
        if (Tx_angles(i) >= 202.5 && Tx_angles(i) < 247.5)
            Tx_sympols(i) = 6;
        end
        if (Tx_angles(i) >= 247.5 && Tx_angles(i) < 292.5)
            Tx_sympols(i) = 7;
        end
        if (Tx_angles(i) >= 292.5 && Tx_angles(i) < 337.5)
            Tx_sympols(i) = 8;
        end
    end
    for i = 1:length(Rx_angles)
        if (Rx_angles(i) >= 0 && Rx_angles(i) < 22.5) || (Rx_angles(i) >= 337.5 && Rx_angles(i) < 360)
            Rx_sympols(i) = 1;
        end
        if (Rx_angles(i) >= 22.5 && Rx_angles(i) < 67.5)
            Rx_sympols(i) = 2;
        end
        if (Rx_angles(i) >= 67.5 && Rx_angles(i) < 112.5)
            Rx_sympols(i) = 3;
        end
        if (Rx_angles(i) >= 112.5 && Rx_angles(i) < 157.5)
            Rx_sympols(i) = 4;
        end
        if (Rx_angles(i) >= 157.5 && Rx_angles(i) < 202.5)
            Rx_sympols(i) = 5;
        end
        if (Rx_angles(i) >= 202.5 && Rx_angles(i) < 247.5)
            Rx_sympols(i) = 6;
        end
        if (Rx_angles(i) >= 247.5 && Rx_angles(i) < 292.5)
            Rx_sympols(i) = 7;
        end
        if (Rx_angles(i) >= 292.5 && Rx_angles(i) < 337.5)
            Rx_sympols(i) = 8;
        end
    end
    incorrecr_bits = sum(Rx_sympols ~= Tx_sympols);
end
%% Mapping to 16QAM
function y = mapping_16QAM(x)
    M = 16;
    bits = log2(M);
    Eb = 2.5; %1;
    E = 0.4 * Eb;
    split_x = reshape(x, bits, length(x)/bits)';
    split_x_indices = bi2de(split_x, 'left-msb')+1;
    y = zeros(1, length(split_x));
    %          [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  12,  13, 14, 1
    R_values = [-3, -3, -3, -3, -1, -1, -1, -1,  3,  3,  3,  3,   1,   1,  1, 1];
    I_values = [-3, -1,  3,  1, -3, -1,  3,  1, -3, -1,  3,  1,  -3,  -1,  3, 1];
    y = sqrt(E)*(R_values(split_x_indices)+1j*I_values(split_x_indices));
end
%% BER for 16QAM
function incorrect_bits = BER_16QAM(Tx_bits, Rx)
    M = 16;
    bits = log2(M);
    Rx_bits = zeros(1, bits * length(Rx));
    k = 1;
    for i = 1:bits:length(Rx_bits)
        if (real(Rx(k)) >= 0)
            Rx_bits(i) = 1;
        else
            Rx_bits(i) = 0;
        end
        if (real(Rx(k)) >= -2 && real(Rx(k)) < 2)
            Rx_bits(i + 1) = 1;
        else
            Rx_bits(i + 1) = 0;
        end
        if (imag(Rx(k)) >= 0)
            Rx_bits(i + 2) = 1;
        else
            Rx_bits(i + 2) = 0;
        end
        if (imag(Rx(k)) >= -2 && imag(Rx(k)) < 2)
            Rx_bits(i + 3) = 1;
        else
            Rx_bits(i + 3) = 0;
        end
        k = k + 1;
    end
    incorrect_bits = sum(Rx_bits ~= Tx_bits);
end
%% BER FSK
function incorrect_bits = BER_FSK(Tx_bits, Rx)
    x1 = real(Rx);
    x2 = imag(Rx);
    y = x1 - x2;
    Rx_bits = 0.5 * (1 - sign(y));
    incorrect_bits = sum(Rx_bits ~= Tx_bits);
end
%% Generate Random Data
function data_transmitted = Generate_Random_Data()
Tb = 7;
Eb = 1;
delta_f = 1/Tb;
t = 0:1:Tb-1;
S1_BB(1:Tb, 1) = sqrt(2*Eb/Tb)*(cos(2*pi*delta_f*t)+1j*sin(2*pi*delta_f*t));
S2_BB(1:Tb, 1) = sqrt(2*Eb/Tb);
% S1_BB(1:Tb, 1) = sqrt(2*Eb/Tb)*(cos(pi*delta_f*t)-1j*sin(pi*delta_f*t));
% S2_BB(1:Tb, 1) = sqrt(2*Eb/Tb)*(cos(pi*delta_f*t)+1j*sin(pi*delta_f*t));
number_of_bits=100;                         % number of bits in each realization
data=randi([0 1],[1,number_of_bits+1]);     % generate 1x100 random numbers '0' or '1'
y(Tb, length(data))= 0;
y(:, find(data==0))= repmat(S1_BB, 1, length(find(data==0)));
y(:, find(data==1))= repmat(S2_BB, 1, length(find(data==1)));
data=reshape(y, 1, size(y, 1)*size(y, 2));
data=data(:);                               % convert data to column vector 700x1
% Generate random time shift
td=randi([0 Tb-1]);                         % generate random number from 0 to 6
% Concatenate with Random Data
data_transmitted = (data(td+1:Tb*number_of_bits+td))';    % window data from td to 700+td (700 samples)
end