clc;clear all;close all;

pr1 = [-1 1;-1 1];
net1 = newp(pr1,1);
net11 = newp(pr1,1);

net1.inputweights{1}.initFcn='rands';
net1.biases{1}.initFcn='rands';
net11.inputweights{1}.initFcn='rands';
net11.biases{1}.initFcn='rands';

index = 0;
while index == 0
    net1=init(net1);
    iw1=net1.IW{1};
    b1=net1.b{1};
    p1=[1 -1 1 -1;1 -1 -1 1];
    [a1,pr]=sim(net1,p1);
    
    net11=init(net11);
    iw11=net11.IW{1};
    b11=net11.b{1};
    p11=[-1 1 1 -1;-1 1 -1 1];
    [a11,pr1]=sim(net11,p11);
    
    pr2=[0 1;0 1];
    net2=newp(pr2,1);
    net2.trainParam.epochs=10;
    net2.trainParam.show=1;
    p21=ones(size(a1));
    p21=p21.*a1;
    p22=ones(size(a11));
    p22=p22.*a11;
    
    p2=[p21;p22];
    t2=[1 1 0 0];
    [net2,tr2]=train(net2,p2,t2);
    epoch2=tr2.epoch;
    perf2=tr2.perf;
    iw2=net2.IW{1};
    b2=net2.b{1};
    a2=sim(net2,p2);
    
    save Percept02 net1 net2
    
    if a2 == t2
        index = 1;
    end
end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    