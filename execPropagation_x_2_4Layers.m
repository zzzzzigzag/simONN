function [ w_1_new,w_2_new,w_3_new,w_4_new,Error_new ] = execPropagation_x_2_4Layers( input,target_output,w_1,w_2,w_3,w_4,b_1,b_2,b_3,b_4,yita )

%% Forward Propagation

h_net = w_1*input + b_1;

% Activate Function: y=x^2
h_out = h_net.^2;%Act2

g_net = w_2*h_out + b_2;
g_out = 1./(1+exp(-g_net));

f_net = w_3*g_out + b_3;
f_out = 1./(1+exp(-f_net));

o_net = w_4*f_out + b_4;
o_out = 1./(1+exp(-o_net));

%% Backward Propagation

delta_o = -(target_output-o_out).*o_out.*(1-o_out);
w_4_new = w_4 - yita*delta_o*f_out.';

delta_f = ((delta_o.')*w_4).'.*f_out.*(1-f_out);
w_3_new = w_3 - yita*delta_f*g_out.';

delta_g = ((delta_f.')*w_3).'.*g_out.*(1-g_out);
w_2_new = w_2 - yita*delta_g*h_out.';

% Activate Function: y=x^2
delta_h = ((delta_g.')*w_2).'.*h_net*2;%Act2
w_1_new = w_1 - yita*delta_h*input.';

Error_new = sum((target_output-o_out).^2)/2;
end
