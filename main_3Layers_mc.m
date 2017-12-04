clc;clear all;close all;

tolerance_high = 1e-4;
tolerance_low = 1e-4;

mc = 0;

%% import training data
load('data2.mat');
SNR_MAX = max(SNR);
BER_MAX = max(BER);
SNR = SNR/SNR_MAX;
data_num = 41;

%% execPropagation: y=sigmoid(x)
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

[w_1_new,w_2_new,w_3_new,Error] =...
    execPropagation_x_2_3Layers(input,target_output,w_1,w_2,w_3,b_1,b_2,b_3,yita);%3Layers
delta_w_3 = w_3_new - w_3;
delta_w_2 = w_2_new - w_2;
delta_w_1 = w_1_new - w_1;
w_3 = w_3_new;
w_2 = w_2_new;
w_1 = w_1_new;
count = count+1;
[input,target_output] = alterIOdata( mod(count,data_num)+1,SNR,BER );
err = [err,Error];

if(Error < tolerance_high)
    success_count = success_count+1;
end
% execute once
% success_count < data_num

while(success_count < data_num)
    while(Error > tolerance_high)
        success_count = 0;
        [w_1,w_2,w_3,Error,...
            delta_w_3,delta_w_2,delta_w_1] =...
            execPropagation_x_2_3Layers_mc(input,target_output,w_1,w_2,w_3,b_1,b_2,b_3,yita,...
            mc,delta_w_3,delta_w_2,delta_w_1);%3Layers
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