%% newp�������������֪��
% *************************************************************************
% ʹ��newp���������֪������һ���������Ȩֵ����ֵ��ѵ���ڶ��㣬��������
% ��һ��ʹ��������Ԫ������Ƚ���������������Ԫ��������������
% *************************************************************************
clear all;close all;clc;
p = [0 0 1 1; 0 1 0 1];
t = repmat([0 1 1 0],8,1);
net1 = newp(minmax(p), 8, 'hardlim', 'learnp'); % �½���һ���֪����3����Ԫ��Ȩֵ����ֵ���
net1.inputWeights{1}.initFcn = 'rands';
net1.biases{1}.initFcn       = 'rands';
i=0;
while i==0
    net1 = init(net1);
    iw1=net1.IW{1};
    b1=net1.b{1};
    a1   = sim(net1, p); % ��һ�������Ϊ�ڶ�������
    net2 = newp(minmax(a1), 1); % �½��ڶ����֪����һ����Ԫ
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

