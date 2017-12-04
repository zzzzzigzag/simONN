clc;clear all;close all;

tolerance_high = 1e-4;
tolerance_low = 1e-4;

%% import training data
load('data2.mat');
SNR_MAX = max(SNR);
BER_MAX = max(BER);
SNR = SNR/SNR_MAX;
data_num = 101;

% %% execPropagation: y=sigmoid(x)
% init_3Layers;   % initialization
% count = 0;
% success_count = 0;
% err = [];
% while(success_count < data_num)
%     while(Error > tolerance_high)
%         success_count = 0;
%         [w_1,w_2,w_3,Error] =...
%             execPropagation_sigmoid_3Layers(input,target_output,w_1,w_2,w_3,b_1,b_2,b_3,yita);%3Layers
%         count = count+1;
%         [input,target_output] = alterIOdata( mod(count,data_num)+1,SNR,BER );
%         err = [err,Error];
%     end
%     success_count = success_count+1;
%     if(length(err)-(data_num-success_count) > 0)
%         Error = err(end - (data_num - success_count));
%     end
% end
% fprintf('Sigmoid: %d times\n',count);
% figure;
% plot(err),xlabel('iterations'),ylabel('error'),title('sigmoid(x)');
% save('weighTrained_sigmoid_3Layers','w_1','w_2','w_3','b_1','b_2','b_3');

%% execPropagation: y=x^2
init_3Layers;   % initialization
count = 0;
success_count = 0;
err = [];

while(success_count < data_num)
    while(Error > tolerance_high)
        success_count = 0;
        [w_1,w_2,w_3,Error] =...
            execPropagation_x_2_3Layers(input,target_output,w_1,w_2,w_3,b_1,b_2,b_3,yita);%3Layers
        count = count+1;
        [input,target_output] = alterIOdata( mod(count,data_num)+1,SNR,BER );
        err = [err,Error];
    end
    success_count = success_count+1;
    if(length(err)-(data_num-success_count) > 0)
        Error = err(end - (data_num - success_count));
    end
end
fprintf('y=x^2: %d times\n',count);
figure;
plot(err),xlabel('iterations'),ylabel('error'),title('y=x^2');
save('weighTrained_x_2_3Layers','w_1','w_2','w_3','b_1','b_2','b_3');

figure;
plot(SNR*SNR_MAX,BER,'r-');
hold on;

h_net = w_1*SNR + repmat(b_1,1,data_num);

% Activate Function: y=x^2
h_out = h_net.^2;%Act2

g_net = w_2*h_out + repmat(b_2,1,data_num);
g_out = 1./(1+exp(-g_net));

o_net = w_3*g_out + repmat(b_3,1,data_num);
o_out = 1./(1+exp(-o_net));

scatter(SNR*SNR_MAX,o_out,'b+');





