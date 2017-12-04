clc;clear all;close all;

tolerance_high = 2.5e-3;
tolerance_low = 1e-4;

load('data.mat');
SNR_MAX = max(SNR);
SNR = SNR/SNR_MAX;
data_num = 41;

%% execPropagation: y=sigmoid(x)
init_4Layers;   % initialization
count = 0;
success_count = 0;
err = [];
while(success_count < data_num)
    while(Error > tolerance_high)
        success_count = 0;
        [w_1,w_2,w_3,w_4,Error] =...
            execPropagation_sigmoid_4Layers(input,target_output,w_1,w_2,w_3,w_4,b_1,b_2,b_3,b_4,yita);%4Layers
        count = count+1;
        [input,target_output] = alterIOdata( mod(count,data_num)+1,SNR,BER );
        err = [err,Error];
    end
    success_count = success_count+1;
    if(length(err)-(data_num-success_count) > 0)
        Error = err(end - (data_num - success_count));
    end
end
fprintf('Sigmoid: %d times\n',count);
figure;
plot(err),xlabel('iterations'),ylabel('error'),title('sigmoid(x)');
save('weighTrained_sigmoid_4Layers','w_1','w_2','w_3','w_4','b_1','b_2','b_3','b_4');
%% execPropagation: y=x^2
init_4Layers;   % initialization
count = 0;
success_count = 0;
err = [];
while(success_count < data_num)
    while(Error > tolerance_high)
        success_count = 0;
        [w_1,w_2,w_3,w_4,Error] =...
            execPropagation_x_2_4Layers(input,target_output,w_1,w_2,w_3,w_4,b_1,b_2,b_3,b_4,yita);%2Layers
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
save('weighTrained_x_2_4Layers','w_1','w_2','w_3','w_4','b_1','b_2','b_3','b_4');
