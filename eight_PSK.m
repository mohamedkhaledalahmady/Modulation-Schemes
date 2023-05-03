clear
clc
Number_of_bits= 999999;
bits = randi ([0 1] ,[1,Number_of_bits]);

%Mapping the bits to 8PSK
symbol_indices = bi2de(reshape(bits, 3, [])', 'left-msb') + 1;
phases = [0 pi/4 3*pi/4 pi/2 7*pi/4 3*pi/2 pi 5*pi/4]; 
eight_PSK_symbols =conj(( exp(1j*phases(symbol_indices)))');

%Add noise
SNR_db=-2:5;
Eb = 1/3 ;
invSNR=10.^(SNR_db./10);
N0=(Eb./invSNR);
noise = sqrt(N0./2).*randn(size(eight_PSK_symbols))+ 1j*sqrt(N0./2).*randn(size(eight_PSK_symbols)) ;
noisy_symbols = eight_PSK_symbols + noise ;
noisy_symbols_used = reshape (noisy_symbols, [],1);

%decode the noisy symbols
decoded_symbols = zeros (Number_of_bits*(8/3) ,1) ;
Unique_matrix = unique(eight_PSK_symbols);
for i = 1 : length(noisy_symbols_used)
Distances = zeros (length(Unique_matrix),1);
for k= 1 : length (Unique_matrix)
Distances(k,1) = abs (noisy_symbols(i)- Unique_matrix(k));
end
[~,min_index] = min (Distances);
decoded_symbols(i,1) =  Unique_matrix(min_index);
end
decoded_symbols = conj((decoded_symbols)');
decoded_symbols = (reshape ((decoded_symbols) , Number_of_bits/3 ,8)); 

%Reobtain the bits from the decoded symbols
bits_recovered = zeros (Number_of_bits ,8);
phase_angles = mod( angle(decoded_symbols), 2*pi);
for m =1:8
[~, symbol_indices] = min(abs(phase_angles(:,m) - phases), [], 2);
bits_recover = (de2bi(symbol_indices-1, 3, 'left-msb'));
bits_recovered(:, m) = reshape((bits_recover )',Number_of_bits , 1);
end
 
%Calculate the number of bit differences
num_diff = zeros (1,8);
for i = 1:size(bits_recovered,2)
diff = bits_recovered(:,i) ~= (bits)'; 
num_diff(i) = sum(diff);
end
error = num_diff ./Number_of_bits;
semilogy(SNR_db,error,'b');
hold on;
semilogy(SNR_db, erfc(sqrt(3*invSNR)*sin(pi/8))/3 , 'r');
hold on;

M = 8; % Modulation order
EbNo = -2:0.5:5; 
ber = berawgn(EbNo, 'psk', M, 'nondiff');
semilogy(EbNo, ber);

xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate');
legend('Simulation', 'Theoretical','exact');
title('Bit Error Rate for 8-PSK modulation');