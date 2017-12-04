%% newp函数建立两层感知器
% *************************************************************************
% 使用newp建立两层感知器，第一层随机生成权值和阈值，训练第二层，可以收敛
% 第一层使用两个神经元收敛会比较慢，可以增加神经元个数，比如三个
% *************************************************************************
clear all;close all;clc;
p = [0 0 1 1; 0 1 0 1];
t = repmat([0 1 1 0],8,1);
net1 = newp(minmax(p), 8, 'hardlim', 'learnp'); % 新建第一层感知器，3个神经元，权值和阈值随机
net1.inputWeights{1}.initFcn = 'rands';
net1.biases{1}.initFcn       = 'rands';
i=0;
while i==0
    net1 = init(net1);
    iw1=net1.IW{1};
    b1=net1.b{1};
    a1   = sim(net1, p); % 第一层输出作为第二层输入
    net2 = newp(minmax(a1), 1); % 新建第二层感知器，一个神经元
    net2.trainParam.epochs = 10;
    % net2.trainParam.show   = 1;
    net2 = train(net2, a1, t);
    iw2=net2.IW{1};
    b2=net2.b{1};
    a2   = sim(net2, a1);
    if a2 == t
        i=1;
    end
end

