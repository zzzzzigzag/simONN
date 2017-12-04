clc;clear all; close all;

xor_function = 0;
IQ_demod = 1;
pam_4 = 0;


%% XOR FUNCTION
if(xor_function)
% load('data_xor.mat');
% input = [A;B];
% target_output = Y;
end

%% IQ DEMOD
if(IQ_demod)
% c = 3e8;
% lambda = 1550e-9;
% carrier_freq = c/lambda;
% sample_rate = 100e9;

load('IQ_1.mat');
load('IQ_2.mat');
IQ_x = downsample(IQ_x,8,4);

% [~,Nsymb] = size(IQ_data);
% t = [0 : 1 / sample_rate : (Nsymb - 1) / sample_rate];
% Carry_I = cos(2 * pi * carrier_freq * t);
% Carry_Q = sin(2 * pi * carrier_freq * t);
% 
% input = real(IQ_data) .* Carry_I - imag(IQ_data) .* Carry_Q;

input = [real(Y_Rx_ss);imag(Y_Rx_ss)];

target_output = [real(IQ_x);imag(IQ_x)];

end
%% PAM4
if(pam_4)
% load pam4
% input = r;
% target_output = x; %matrixrepeat(x,4);  matrixrepeat([-3,-1,1,3],5000);  linspace(-3,3,20000);  repmat([-3,-1,1,3],1,5000);
end
%% TRAIN

x1 = input(:,1:2:end-1);
y1 = target_output(:,1:2:end-1); 

Epochs = 1000;
net = newff(x1,y1,[16,8,8],{'tansig','tansig','tansig','tansig'},'trainlm');
net.trainParam.epochs = Epochs;  
net.trainParam.goal = 1e-7;
net.trainParam.min_grad = 1e-20;
net.trainParam.show = 200;
net.trainParam.time = inf;
net.trainParam.max_fail = 1000;   %Validation Checks
net = init(net);

[net,tr] = train(net,x1,y1);

%% Valuing

out = sim(net,x1);
%MSE = mse(out-y1);

%% TRAIN PERFORMANCE

% figure;
% plot(x1,y1,'b-',x1,out,'r*'),legend('Theoretical','Training Data'),title('Training Performance'),xlabel('SNR/dB'),ylabel('BER');

if(IQ_demod)
    figure;
    plot(out(1,:)+1j*out(2,:),'r*');
end
if(pam_4)
% figure;
% plot(1:length(x1),x1,'bo',1:length(out),out,'r*'),legend('Theoretical','Training Data'),title('Training Performance');
end
%% SAVE WEIGHS

wb = formwb(net,net.b,net.iw,net.lw);
[b,iw,lw] = separatewb(net,wb);

save('weighTrained_3Layers_toolbox','b','iw','lw');

%% FITTING

x2 = input(:,2:2:end);
y2 = target_output(:,2:2:end);
f_out = sim(net,x2);

%% FITTING PERFORMANCE
 
% figure;
% plot(x2,y2,'b-',x2,f_out,'r*'),legend('Theoretical','Fitting Data'),title('Fitting Performance'),xlabel('SNR/dB'),ylabel('BER');
if(IQ_demod)
    figure;
    plot(f_out(1,:)+1j*f_out(2,:),'go');
end
if(pam_4)
% figure;
% plot(1:length(y2),y2,'bo',1:length(f_out),f_out,'r*'),legend('Theoretical','Fitting Data'),title('Fitting Performance');
end