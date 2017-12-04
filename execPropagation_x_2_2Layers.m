function [ w_1_new,w_2_new,Error_new ] = execPropagation_x_2_2Layers( input,target_output,w_1,w_2,b_1,b_2,yita )

%% Forward Propagation

h_net = w_1*input + b_1;

% Activate Function: y=x^2
h_out = h_net.^2;%Act2

o_net = w_2*h_out + b_2;
o_out = 1./(1+exp(-o_net));

%% Backward Propagation

delta_o = -(target_output-o_out).*o_out.*(1-o_out);
w_2_new = w_2 - yita*delta_o*(h_out.');

% Activate Function: y=x^2
delta_h = ((delta_o.')*w_2).'.*h_net*2;%Act2
w_1_new = w_1 - yita*delta_h*input.';

Error_new = sum((target_output-o_out).^2)/2;
end
