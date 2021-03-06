clc;clear all;close all;

U = eye(4);

syms theta1 theta2 theta3 theta4 theta5 theta6 fai1 fai2 fai3 fai4 fai5 fai6...
     u11_1 u11_2 u11_3 u11_4 u11_5 u11_6...
     u12_1 u12_2 u12_3 u12_4 u12_5 u12_6...
     u21_1 u21_2 u21_3 u21_4 u21_5 u21_6...
     u22_1 u22_2 u22_3 u22_4 u22_5 u22_6 ;

%  sol1 = solve(...
%     'u11_1 * (u11_3*u11_6 + u21_3*u11_5*u12_6) + u21_1 * (u11_2 * (u12_3*u11_6 + u22_3*u11_5*u12_6) + u21_2*u11_4*u12_5*u12_6) = 1', ...
%     'u12_1 * (u11_3*u11_6 + u21_3*u11_5*u12_6) + u22_1 * (u11_2 * (u12_3*u11_6 + u22_3*u11_5*u12_6) + u21_2*u11_4*u12_5*u12_6) = 0', ...
%     'u12_2 * (u12_3*u11_6 + u22_3*u11_5*u12_6) + u22_2 * u11_4 * u12_5 * u12_6 = 0', ...
%     'u12_4 * u12_5 * u12_6 = 0', ...
%     'u11_1 * (u11_3*u21_6 + u21_3*u11_5*u22_6) + u21_1 * (u11_2 * (u12_3*u21_6 + u22_3*u11_5*u22_6) + u21_2*u11_4*u12_5*u22_6) = 0', ...
%     'u12_1 * (u11_3*u21_6 + u21_3*u11_5*u22_6) + u22_1 * (u11_2 * (u12_3*u21_6 + u22_3*u11_5*u22_6) + u21_2*u11_4*u12_5*u22_6) = 1', ...
%     'u12_2 * (u12_3*u21_6 + u22_3*u11_5*u22_6) + u22_2 * u11_4 * u12_5 * u22_6 = 0', ...
%     'u12_4 * u12_5 * u22_6 = 0', ...
%     'u11_1 * u21_3 * u21_5 + u21_1 * (u11_2*u22_3*u21_5 + u21_2*u11_4*u22_5) = 0', ...
%     'u12_1 * u21_3 * u21_5 + u22_1 * (u11_2*u22_3*u21_5 + u21_2*u11_4*u22_5) = 0', ...
%     'u12_2 * u22_3 * u21_5 + u22_2 * u11_4 * u22_5 = 1', ...
%     'u12_4 * u22_5 = 0', ...
%     'u21_1 * u21_2 * u21_4 = 0', ...
%     'u22_1 * u21_2 * u21_4 = 0', ...
%     'u22_2 * u21_4 = 0', ...
%     'u22_4 = 1', ...
%     ...
%     'u12_1 * (u11_3*u11_6 + u21_3*u11_5*u12_6) + u22_1 * (u11_2 * (u12_3*u11_6 + u22_3*u11_5*u12_6) + u21_2*u11_4*u12_5*u12_6) = (u11_1 * (u11_3*u21_6 + u21_3*u11_5*u22_6) + u21_1 * (u11_2 * (u12_3*u21_6 + u22_3*u11_5*u22_6) + u21_2*u11_4*u12_5*u22_6))''  ', ...
%     'u12_2 * (u12_3*u11_6 + u22_3*u11_5*u12_6) + u22_2 * u11_4 * u12_5 * u12_6 = (u11_1 * u21_3 * u21_5 + u21_1 * (u11_2*u22_3*u21_5 + u21_2*u11_4*u22_5))'' ', ...
%     ...u12_2 * (u12_3*u21_6 + u22_3*u11_5*u22_6) + u22_2 * u11_4 * u12_5 * u22_6 - (u12_1 * u21_3 * u21_5 + u22_1 * (u11_2*u22_3*u21_5 + u21_2*u11_4*u22_5))', ...
%     ...u12_4 * u12_5 * u22_6 - (u22_1 * u21_2 * u21_4)', ...
%     ...u12_4 * u22_5 - (u22_2 * u21_4)', ...
%     ...u12_4 * u12_5 * u12_6 - (u21_1 * u21_2 * u21_4)', ...
%     ...
%     'u12_1 = u21_1'' ', ...
%     'u12_2 = u21_2'' ', ...
%     'u12_3 = u21_3'' ', ...
%     'u12_4 = u21_4'' ', ...
%     'u12_5 = u21_5'' ', ...
%     'u12_6 = u21_6'' ');

sol = solve(...
    u11_1 * (u11_3*u11_6 + u21_3*u11_5*u12_6) + u21_1 * (u11_2 * (u12_3*u11_6 + u22_3*u11_5*u12_6) + u21_2*u11_4*u12_5*u12_6) - U(1,1), ...
    u12_1 * (u11_3*u11_6 + u21_3*u11_5*u12_6) + u22_1 * (u11_2 * (u12_3*u11_6 + u22_3*u11_5*u12_6) + u21_2*u11_4*u12_5*u12_6) - U(1,2), ...
    u12_2 * (u12_3*u11_6 + u22_3*u11_5*u12_6) + u22_2 * u11_4 * u12_5 * u12_6 - U(1,3), ...
    u12_4 * u12_5 * u12_6 - U(1,4), ...
    u11_1 * (u11_3*u21_6 + u21_3*u11_5*u22_6) + u21_1 * (u11_2 * (u12_3*u21_6 + u22_3*u11_5*u22_6) + u21_2*u11_4*u12_5*u22_6) - U(2,1), ...
    u12_1 * (u11_3*u21_6 + u21_3*u11_5*u22_6) + u22_1 * (u11_2 * (u12_3*u21_6 + u22_3*u11_5*u22_6) + u21_2*u11_4*u12_5*u22_6) - U(2,2), ...
    u12_2 * (u12_3*u21_6 + u22_3*u11_5*u22_6) + u22_2 * u11_4 * u12_5 * u22_6 - U(2,3), ...
    u12_4 * u12_5 * u22_6 - U(2,4), ...
    u11_1 * u21_3 * u21_5 + u21_1 * (u11_2*u22_3*u21_5 + u21_2*u11_4*u22_5) - U(3,1), ...
    u12_1 * u21_3 * u21_5 + u22_1 * (u11_2*u22_3*u21_5 + u21_2*u11_4*u22_5) - U(3,2), ...
    u12_2 * u22_3 * u21_5 + u22_2 * u11_4 * u22_5 - U(3,3), ...
    u12_4 * u22_5 - U(3,4), ...
...%     u21_1 * u21_2 * u21_4 - U(4,1), ...
...%     u22_1 * u21_2 * u21_4 - U(4,2), ...
...%     u22_2 * u21_4 - U(4,3), ...
...%     u22_4 - U(4,4), ...
    ...
    ...u12_1 * (u11_3*u11_6 + u21_3*u11_5*u12_6) + u22_1 * (u11_2 * (u12_3*u11_6 + u22_3*u11_5*u12_6) + u21_2*u11_4*u12_5*u12_6) - (u11_1 * (u11_3*u21_6 + u21_3*u11_5*u22_6) + u21_1 * (u11_2 * (u12_3*u21_6 + u22_3*u11_5*u22_6) + u21_2*u11_4*u12_5*u22_6))', ...
    ...u12_2 * (u12_3*u11_6 + u22_3*u11_5*u12_6) + u22_2 * u11_4 * u12_5 * u12_6 - (u11_1 * u21_3 * u21_5 + u21_1 * (u11_2*u22_3*u21_5 + u21_2*u11_4*u22_5))', ...
    ...u12_2 * (u12_3*u21_6 + u22_3*u11_5*u22_6) + u22_2 * u11_4 * u12_5 * u22_6 - (u12_1 * u21_3 * u21_5 + u22_1 * (u11_2*u22_3*u21_5 + u21_2*u11_4*u22_5))', ...
    ...u12_4 * u12_5 * u22_6 - (u22_1 * u21_2 * u21_4)', ...
    ...u12_4 * u22_5 - (u22_2 * u21_4)', ...
    ...u12_4 * u12_5 * u12_6 - (u21_1 * u21_2 * u21_4)', ...
    ...
    exp(1i*fai1)*(exp(1i*theta1)-1) - 2*u11_1,...
    exp(1i*fai2)*(exp(1i*theta2)-1) - 2*u11_2,...
    exp(1i*fai3)*(exp(1i*theta3)-1) - 2*u11_3,...
    exp(1i*fai4)*(exp(1i*theta4)-1) - 2*u11_4,...
    exp(1i*fai5)*(exp(1i*theta5)-1) - 2*u11_5,...
    exp(1i*fai6)*(exp(1i*theta6)-1) - 2*u11_6,...  
    ...
    1i*exp(1i*fai1)*(1+exp(1i*theta1)) - 2*u12_1,...
    1i*exp(1i*fai2)*(1+exp(1i*theta2)) - 2*u12_2,...
    1i*exp(1i*fai3)*(1+exp(1i*theta3)) - 2*u12_3,...
    1i*exp(1i*fai4)*(1+exp(1i*theta4)) - 2*u12_4,...
    1i*exp(1i*fai5)*(1+exp(1i*theta5)) - 2*u12_5,...
    1i*exp(1i*fai6)*(1+exp(1i*theta6)) - 2*u12_6,...
    ...
    1i*(exp(1i*theta1)+1) - 2*u21_1,...
    1i*(exp(1i*theta2)+1) - 2*u21_2,...
    1i*(exp(1i*theta3)+1) - 2*u21_3,...
    1i*(exp(1i*theta4)+1) - 2*u21_4,...
    1i*(exp(1i*theta5)+1) - 2*u21_5,...
    1i*(exp(1i*theta6)+1) - 2*u21_6,...
    ...
    1 - exp(1i*theta1) - 2*u22_1,...
    1 - exp(1i*theta2) - 2*u22_2,...
    1 - exp(1i*theta3) - 2*u22_3,...
    1 - exp(1i*theta4) - 2*u22_4,...
    1 - exp(1i*theta5) - 2*u22_5,...
    1 - exp(1i*theta6) - 2*u22_6);...
    ...
%     u12_1 - u21_1', ...
%     u12_2 - u21_2', ...
%     u12_3 - u21_3', ...
%     u12_4 - u21_4', ...
%     u12_5 - u21_5', ...
%     u12_6 - u21_6');
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
