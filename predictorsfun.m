function [R2,MSE,RMSE,MAPE,MAD]=predictorsfun(y,y1)


%% 计算各种预测数据的指标,可决系数,误差平方和等
% 计算可决系数,mse等
% R2=1-sum((y-y1).^2)/sum((y-mean(y)).^2);%可决系数
Q=sum((y-y1).^2);
R2=1-sqrt(Q/sum(y.^2));% 拟合优度
MSE=sum((y-y1).^2)/(length(y)-1);% 均方误差
RMSE=sqrt(MSE);% 均方根误差
MAPE=sum(abs(y1-y)*100./y)/length(y);% 相对百分误差
MAD=mean(abs(y1-y));% 平均绝对误差

