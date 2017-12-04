function [ w_1_new,w_2_new,w_3_new,Error_new,...
    delta_w_3_new,delta_w_2_new,delta_w_1_new...
    ] = execPropagation_x_2_3Layers_mc( input,target_output,w_1,w_2,w_3,b_1,b_2,b_3,yita,...
    mc,delta_w_3,delta_w_2,delta_w_1 )

%% Forward Propagation

h_net = w_1*input + b_1;

% Activate Function: y=x^2
h_out = h_net.^2;%Act2

g_net = w_2*h_out + b_2;
g_out = 1./(1+exp(-g_net));

o_net = w_3*g_out + b_3;
o_out = 1./(1+exp(-o_net));

%% Backward Propagation

delta_o = -(target_output-o_out).*o_out.*(1-o_out);
w_3_new = w_3 - (1-mc)*yita*delta_o*g_out.' - mc*delta_w_3;
delta_w_3_new = w_3_new - w_3;

delta_g = ((delta_o.')*w_3).'.*g_out.*(1-g_out);
w_2_new = w_2 - (1-mc)*yita*delta_g*h_out.' - mc*delta_w_2;
delta_w_2_new = w_2_new - w_2;

% Activate Function: y=x^2
delta_h = ((delta_g.')*w_2).'.*h_net*2;%Act2
w_1_new = w_1 - (1-mc)*yita*delta_h*input.' - mc*delta_w_1;
delta_w_1_new = w_1_new - w_1;

Error_new = sum((target_output-o_out).^2)/2;
end
