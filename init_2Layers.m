input = SNR(1);
target_output = BER(1);

% rand
w_1 = rand(8,1);
w_2 = rand(1,8);
b_1 = rand(8,1);
b_2 = rand(1,1);
yita = 0.5;
Error = sum((target_output-input).^2)/2;
