clear
clc
number_of_realizations = 500;
number_of_bits = 101;
number_of_samples = 7;
n = 700;
Eb= 1;
Tb= 7;
bits = randi ([0 1] ,number_of_realizations ,number_of_bits);

% Data = zeros (number_of_realizations , size(bits,2)*number_of_samples);
Data = [];
s_one = zeros(1, number_of_samples);
s_one (1,:) = sqrt(2*Eb/Tb);
s_zero = (sqrt(2*Eb/Tb))*cos(2*pi*(0:Tb/number_of_samples:(Tb-Tb/number_of_samples))/Tb)+1j*(sqrt(2*Eb/Tb))*sin(2*pi*(0:Tb/number_of_samples:(Tb-Tb/number_of_samples))/Tb);
 for row = 1: number_of_realizations
     for column = 1 :number_of_bits
         if bits(row , column ) == 1
             Data = [Data , s_one];
         else
            Data = [Data, s_zero];
         end
     end   
 end
 Data = (reshape (Data, number_of_bits*Tb , number_of_realizations))';
 Data_transmitted = zeros (number_of_realizations, n);
 
 for i = 1: number_of_realizations
     td=randi(number_of_samples)-1;              % generate random number from 0 to 6
     X = Data(i,:);
     Data_transmitted(i,:) = X(td+1: n+td);
 end
 
 average=zeros(1,n);  
for i = 1 : n
    average(1,i) = mean(conj(Data_transmitted(:,1)) .* Data_transmitted(:,i));
end
average=[fliplr(average(1, 2:end)) average];
PSD = fftshift(fft(average));
N = length(PSD);                                                                      
Fs=7;                                        
freq = (-N/2+1:N/2)*(Fs/(N));                   
figure
plot(freq, abs(PSD));                 