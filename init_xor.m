input = [A(1);B(1)];
target_output = Y(1);

% rand
w_1 = rand(8,2);
w_2 = rand(8,8);
w_3 = rand(1,8);
b_1 = rand(8,1);
b_2 = rand(8,1);
b_3 = rand(1,1);
yita = 0.5;
Error = sum((target_output-input).^2)/2;