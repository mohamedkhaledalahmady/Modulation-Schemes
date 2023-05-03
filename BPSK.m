clear
clc
Number_of_bits= 1e6;
bits = randi ([0 1] ,[1,Number_of_bits]);

%Mapping the bits to BPSK
BPSK_symbols = ((2*bits)-1)';

%Add noise
SNR_db=-2:0.5:5;
Eb = 1 ;
invSNR=10.^(SNR_db./10);
N0=(1./invSNR);
noise = sqrt(N0./2).*randn(size(BPSK_symbols)) ;
noisy_symbols = BPSK_symbols + noise ;
noisy_symbols_used = reshape (noisy_symbols, [],1);

%Decode the noisy symbols 
decoded_symbols = zeros ( Number_of_bits*length(SNR_db) , 1);
Unique_matrix = unique(BPSK_symbols);
for i = 1 : length(noisy_symbols_used)
Distances = zeros (length(Unique_matrix),1);
for k= 1 : length (Unique_matrix)
Distances(k,1) = abs (noisy_symbols(i)- Unique_matrix(k));
end
[~,min_index] = min (Distances);
decoded_symbols(i,1) =  Unique_matrix(min_index);
end
decoded_symbols= (decoded_symbols)';
decoded_symbols = (reshape (decoded_symbols , Number_of_bits ,length (SNR_db)));
num_diff = zeros(1,size(decoded_symbols,2));

%Calculte the number of bit differences
for i = 1:size(decoded_symbols,2)
diff = decoded_symbols(:,i) ~= BPSK_symbols ;
num_diff(i) = sum(diff);
end

%Plot the BER
error = num_diff ./length (BPSK_symbols);
semilogy(SNR_db,error,'b');
hold on;
semilogy(SNR_db, 0.5.*erfc(sqrt(Eb./N0)) , 'r');
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate');
legend('Simulation', 'Theory');
title('Bit Error Rate for QPSK modulation');