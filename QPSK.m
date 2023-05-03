clear
clc
Number_of_bits = 1e6;
bits = randi ([0 1] ,[1,Number_of_bits]);

%Mapping the bits to QPSK
symbols = ((2*reshape(bits , 2, [])-1))';
QPSK_symbols = symbols (:,1)+1j*symbols(:,2); 

%Add noise
SNR_db=-2:0.5:5;
Eb = 1;
invSNR=10.^(SNR_db./10);
N0=(Eb./invSNR);
noise = sqrt(N0./2).*randn(size(QPSK_symbols))+ 1j*sqrt(N0./2).*randn(size(QPSK_symbols)) ;
noisy_symbols = QPSK_symbols + noise ;
noisy_symbols_used = reshape (noisy_symbols, [],1);

%decode the noisy symbols
decoded_symbols = zeros ((Number_of_bits*length(SNR_db))/2,1);
Unique_matrix = unique(QPSK_symbols);
for i = 1 : length(noisy_symbols_used)
Distances = zeros (length(Unique_matrix),1);
for k= 1 : length (Unique_matrix)
Distances(k,1) = abs (noisy_symbols(i)- Unique_matrix(k));
end
[~,min_index] = min (Distances);
decoded_symbols(i,1) = Unique_matrix(min_index) ;
end
decoded_symbols = conj((decoded_symbols)');
decoded_symbols = (reshape ((decoded_symbols) , Number_of_bits/2 , length(SNR_db)));

%Reobtain the bits from the decoded symbols
[m, n] = size(decoded_symbols); 
B = zeros(m, 2*n); 
for j = 1:n
    B(:,2*j-1) = real(decoded_symbols(:,j));
    B(:,2*j) = imag(decoded_symbols(:,j));
end
B = reshape(B , Number_of_bits ,length(SNR_db) );
C = reshape (symbols , Number_of_bits, 1);
num_diff = zeros(1,size(B,2));
for i = 1:size(B,2)
diff = B(:,i) ~= C; 
num_diff(i) = sum(diff);
end

%plot
error = num_diff ./Number_of_bits;
semilogy(SNR_db,error,'b')
hold on
semilogy(SNR_db, 0.5.*erfc(sqrt(invSNR)) , 'r')
hold on
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate');
legend('Simulation', 'Theory');
title('Bit Error Rate for QPSK modulation');