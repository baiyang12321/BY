clc;clear;
%%采用了FCM算法进行权重选择。
%%FCM聚类时，包含训练数据和预测数据。
%%针对小样本数据，采用留一交叉验证法进行训练
clc;clear;warning off;
%删除工作文件夹
if exist('workfile','dir')~=0
    rmdir('workfile', 's');
end
mkdir('workfile');%%创造一个文件夹
addpath('workfile');%%将该文件夹添加到路径
try
    [adata_train,bdata_train,cdata_train]=xlsread('train.xlsx');
    [adata_forecast,bdata_forecast,cdata_forecast]=xlsread('prediction.xlsx');
catch
    errordlg('打开文件失败','Wrong File');
    return
end

data_train=adata_train;
P_column=str2num('2 3 4 5 6 8');%训练输入数据2 3 4 5 6 7 8 9 10
T_column=str2double('9');%标签
data_forecast=adata_forecast(:,P_column);
data_forecast_Depth=adata_forecast(:,1);

dataX=data_train(:,P_column);
dataY=data_train(:,T_column);
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%空值全部为零

if exist('委员会机器输入数据集.xlsx','file')~=0%%删掉已存在的输出文件
    delete('委员会机器输入数据集.xlsx')
end
xlswrite('委员会机器输入数据集.xlsx',dataX,'inputdata');
xlswrite('委员会机器输入数据集.xlsx',dataY,'outputdata');

%训练数据归一化
data_total=[dataX;data_forecast];
[dataX_total_anormalization ,inputps]=mapminmax(data_total');%归一化默认为-1~1
dataX_total_anormalization=dataX_total_anormalization';
data_train_anormalization=dataX_total_anormalization(1:size(dataX,1),:);%训练数据归一化结果
data_forecast_anormalization=dataX_total_anormalization(size(dataX,1)+1:end,:);%预测数据归一化结果
%K折交叉验证
K_fold=1;%折数
% K_fold=size(dataY,1);%留一交叉验证
indices =crossvalind('Kfold', size(dataX,1), K_fold);
train_P_data=cell(1,K_fold);%%预定义变量，分配内存
train_T_data=cell(1,K_fold);%%预定义变量，分配内存
test_P_data=cell(1,K_fold);%%预定义变量，分配内存
test_T_data=cell(1,K_fold);%%预定义变量，分配内存
for i = 1:K_fold %循环多次，分别取出第i部分作为测试样本，其余两部分作为训练样本
    cross_num_test = (indices == i);%提出一个验证集，其余均为训练集
    cross_num_train = ~cross_num_test;
    train_P_data{i} = data_train_anormalization(cross_num_train, :);%提取的训练集
    train_T_data{i} = dataY(cross_num_train, :);
    test_P_data{i} = data_train_anormalization(cross_num_test, :);%提取的验证集
    test_T_data{i} = dataY(cross_num_test, :);
end

[dataX_m,~]=size(dataX);
%%FCM-cluster
cluster_n=2;%类别数
options = [2;500;1e-5;0];
data=[dataX,dataY];

[center_total, U_total, obj_fcn_total] = fcm(dataX_total_anormalization, cluster_n, options);
U_Core=U_total(:,1:dataX_m);
U_Prediction=U_total(:,dataX_m+1:end);

[Fcma_Core,Fcmb_Core]=max(U_Core);%%岩心数据实际分类
Fcmb_Core=Fcmb_Core';
[Fcma_Prediction,Fcmb_Prediction]=max(U_Prediction);%%预测数据实际分类
Fcmb_Prediction=Fcmb_Prediction';

%FCM预测集和岩心数据关系
FCM_figure(dataX_total_anormalization,Fcmb_Core,Fcmb_Prediction);

save Input_data data_train adata_forecast data_forecast_Depth data_train_anormalization data_forecast_anormalization ...
    train_P_data train_T_data test_P_data test_T_data dataY inputps;
save Fcm_data Fcma_Core Fcmb_Core Fcma_Prediction Fcmb_Prediction K_fold U_Core U_Prediction cluster_n;
movefile('Input_data.mat', 'workfile');
movefile('Fcm_data.mat', 'workfile');
%% Elman神经网络
clear;
%%导入数据
load Input_data train_P_data train_T_data test_P_data test_T_data data_train_anormalization dataY;
load Fcm_data K_fold;
%%网络的建立和训练
% 利用循环，设置不同的隐藏层神经元个数
nn=10:2:30;
Elman_cross_RE=zeros(length(nn),K_fold);
for k=1:K_fold
    Re_train=zeros(length(nn),1);
    Re_test=zeros(length(nn),1);
    for i=1:length(nn)
        % 建立Elman神经网络 隐藏层为nn(i)个神经元
        
        net=newelm(minmax(train_P_data{k}'),[nn(i),1],{'tansig','purelin'},'traingdm');
        % 设置网络训练参数
        net.trainparam.epochs = 200;
        net.trainParam.lr = 0.01;
        net.trainParam.mc = 0.9;
        net.trainParam.goal = 1e-2;
        net.trainparam.show = 50;
        net.trainParam.showWindow = false; % 不显示训练窗口
        % Elman网络训练
        net=train(net,train_P_data{k}',train_T_data{k}','useGPU','no');
        % 预测数据
        ty_train=sim(net,train_P_data{k}')';
        ty_test=sim(net,test_P_data{k}')';
        % 计算误差
        %     test_out=mapminmax('reverse',ty,outputps);
        Re_train(i,:)=mean(abs(ty_train-train_T_data{k})./train_T_data{k});
        Re_test(i,:)=mean(abs(ty_test-test_T_data{k})./test_T_data{k});
        % 初始化网络
        net=init(net);
    end
    Elman_cross_RE(:,k)=Re_train+Re_test;%记录每K折训练和验证误差和
end
Elman_cross_RE_sum=sum(Elman_cross_RE,2);
[~,Elman_cross_parameter]=min(Elman_cross_RE_sum);
Elman_hidnumber=nn(Elman_cross_parameter);%得到误差和最小的参数

% 初始化网络
init(net);
%%正式构建网络
net=newelm(minmax(data_train_anormalization'),[Elman_hidnumber,1],{'tansig','purelin'},'traingdm');
% 设置网络训练参数
net.trainparam.epochs = 2000;%迭代次数（训练次数）
net.trainParam.lr = 0.001;%学习率
net.trainParam.mc = 0.9;%附加动量因子
net.trainParam.goal = 1e-3;%训练目标最小误差
net.trainparam.show = 20;%现实频率，这里设置为每训练20次显示一次
net.trainParam.showWindow = false; % 不显示训练窗口
net.trainparam.min_fail=5;% 最大确认失败次数
net.trainparam.min_grad=1e-6;% 最小梯度

% Elman网络训练
[net,tr,Y,E]=train(net,data_train_anormalization',dataY','useGPU','no');
% 预测数据
ybptest=sim(net,data_train_anormalization')';
% 计算误差
ElmanO=ybptest;
Elman_RE=abs(ElmanO-dataY)./dataY;
Elman_ave=mean(Elman_RE);
Expert_RE=Elman_RE;%%每个测试样本的误差
Expert_RE_ace=Elman_ave;%%所有测试样本的平均误差

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save netElman net Elman_hidnumber ElmanO;
movefile('netElman.mat', 'workfile');

%% 极限学习机回归拟合
clearvars -except Expert_RE_ace Expert_RE ;
%%导入数据
load Input_data train_P_data train_T_data test_P_data test_T_data data_train_anormalization dataY;
load Fcm_data K_fold;

% 利用循环，设置不同的参数
nn=2:2:40;
ELM_cross_RE=zeros(length(nn),K_fold);
for k=1:K_fold
    for i=1:length(nn)
        
        [IW,B,LW,TF,TYPE] = elmtrain(train_P_data{k}',train_T_data{k}',nn(i),'sig',0);
        %%ELM仿真测试
        ty_train=elmpredict(train_P_data{k}',IW,B,LW,TF,TYPE)';
        ty_test=elmpredict(test_P_data{k}',IW,B,LW,TF,TYPE)';
        
        % 计算误差
        Re_train(i,:)=mean(abs(ty_train-train_T_data{k})./train_T_data{k});
        Re_test(i,:)=mean(abs(ty_test-test_T_data{k})./test_T_data{k});
        clear IW B LW TF TYPE;
        
    end
    ELM_cross_RE(:,k)=Re_train+Re_test;%记录每K折训练和验证误差和
end
ELM_cross_RE_sum=sum(ELM_cross_RE,2);
[~,ELM_cross_parameter]=min(ELM_cross_RE_sum);
ELM_M=nn(ELM_cross_parameter);%得到误差和最小的参数

% 初始化网络
clear IW B LW TF TYPE;
%%正式构建极限学习机
[IW,B,LW,TF,TYPE] = elmtrain(data_train_anormalization',dataY',ELM_M,'sig',0);

%%ELM仿真测试
ybptest=elmpredict(data_train_anormalization',IW,B,LW,TF,TYPE)';

% 计算误差
ELMO=ybptest;
ELM_RE=abs(ELMO-dataY)./dataY;
ELM_ave=mean(ELM_RE);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save netELM IW B LW TF TYPE ELM_M ELMO;
movefile('netELM.mat', 'workfile');

Expert_RE=[Expert_RE,ELM_RE];%%每个测试样本的误差
Expert_RE_ace=[Expert_RE_ace;ELM_ave];%%所有测试样本的平均误差

%% GRNN神经网络
clearvars -except Expert_RE_ace Expert_RE ;
%%导入数据
load Input_data train_P_data train_T_data test_P_data test_T_data data_train_anormalization dataY;
load Fcm_data K_fold;

% 利用循环，寻找最佳参数
nn=0.3:0.1:0.9;
GRNN_cross_RE=zeros(length(nn),K_fold);
for k=1:K_fold
    for i=1:length(nn)
        % 建立Elman神经网络 传播速度为nn(i)
        net=newgrnn(train_P_data{k}',train_T_data{k}',nn(i));
        % 预测数据
        ty_train=sim(net,train_P_data{k}')';
        ty_test=sim(net,test_P_data{k}')';
        % 计算误差
        Re_train(i,:)=mean(abs(ty_train-train_T_data{k})./train_T_data{k});
        Re_test(i,:)=mean(abs(ty_test-test_T_data{k})./test_T_data{k});
        % 初始化网络
        net=init(net);
    end
    GRNN_cross_RE(:,k)=Re_train+Re_test;%记录每K折训练和验证误差和
end
GRNN_cross_RE_sum=sum(GRNN_cross_RE,2);
[~,GRNN_cross_parameter]=min(GRNN_cross_RE_sum);
GRNN_Spread=nn(GRNN_cross_parameter);%得到误差和最小的参数

% 初始化网络
init(net);
%%正式构建网络GRNN网络
net=newgrnn(data_train_anormalization',dataY',GRNN_Spread);
% 预测数据
ybptest=sim(net,data_train_anormalization')';

% 计算误差
GRNNO=ybptest;
GRNN_RE=abs(GRNNO-dataY)./dataY;
GRNN_ave=mean(GRNN_RE);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save netGRNN net GRNN_Spread GRNNO;
movefile('netGRNN.mat', 'workfile');

Expert_RE=[Expert_RE,GRNN_RE];%%每个测试样本的误差
Expert_RE_ace=[Expert_RE_ace;GRNN_ave];%%所有测试样本的平均误差

%% 委员会机器组合器Train
clearvars -except Expert_RE_ace Expert_RE ;
%%读取数据
load netElman ElmanO;load netELM ELMO;load netGRNN GRNNO;load Input_data dataY;
load Fcm_data Fcmb_Core U_Core cluster_n;

%寻找最好的专家
for cluster_i=1:cluster_n
    
    Expert_FMC_index=find(Fcmb_Core==cluster_i)';
    if size(Expert_FMC_index,2)==0
        continue
    else
        Expert_error=Expert_RE(Expert_FMC_index,:);
        Expert_error_average=mean(Expert_error,1);
        [~,FCM_result_good]=min(Expert_error_average);%%最好的专家
        [~,FCM_result_bad]=max(Expert_error_average);%%最坏的专家
        FCM_result_med=median(Expert_error_average,2);%%良好的专家
        FCM_result_med=find(Expert_error_average==FCM_result_med);
        
        U_Expert0=U_Core(:,Expert_FMC_index);
        [U_Expert_max,~]=max(U_Expert0);
        [U_Expert_min,~]=min(U_Expert0);
        U_Expert_med=median(U_Expert0);
        U_Expert(FCM_result_good,:)=U_Expert_max;
        
        if cluster_n==3%聚两类时，舍弃一个最差的专家(目前只支持两类或三类)
            U_Expert(FCM_result_med,:)=U_Expert_med;
            U_Expert(FCM_result_bad,:)=U_Expert_min;
        else
            U_Expert(FCM_result_bad,:)=0;
            U_Expert(FCM_result_med,:)=U_Expert_min;
        end
        
        ElmanO_Expert=ElmanO(Expert_FMC_index,:);
        ELMO_Expert=ELMO(Expert_FMC_index,:);
        GRNNO_Expert=GRNNO(Expert_FMC_index,:);
        Three_Expert_Out=[ElmanO_Expert,ELMO_Expert,GRNNO_Expert];
        
        CM_Expert=Three_Expert_Out*U_Expert;
        CM_Expert=diag(CM_Expert);
        CMO(Expert_FMC_index,:)=CM_Expert;
    end
    FCM_result{cluster_i}=[FCM_result_good,FCM_result_med,FCM_result_bad];
    clear Expert_FMC_index Expert_error U_Expert0 U_Expert_max U_Expert_min U_Expert_med ...
        U_Expert ElmanO_Expert ELMO_Expert GRNNO_Expert Three_Expert_Out CM_Expert;
    
    
end

CM_RE=abs(CMO-dataY)./dataY;
CM_ave=mean(CM_RE);

save FCM_Combiner_result  FCM_result;
movefile('FCM_Combiner_result.mat', 'workfile');
disp('各专家相对误差:')
disp(Expert_RE_ace)
disp('委员会机器相对误差:')
disp(CM_ave)
%%%%%%%%%@%%%%%%%%%@%%%
%%%%%%%%@%%%%%%%%%@%%%%
%%%%%%%@%%%%%%%%%@%%%%%
%%%%%%@%%%%%%%%%@%%%%%%
%%%%%@%%%%%%%%%@%%%%%%%
%%%%@%%%%%%%%%@%%%%%%%%
%%%@%%%%%%%%%@%%%%%%%%%
%% forcast
%%Elman仿真
clearvars -except Expert_RE_ace Expert_RE ;
load Input_data data_forecast_anormalization;
load netElman net;
ybptest=sim(net,data_forecast_anormalization')';
ElmanO=ybptest;
save Elmanoutput_forecast ElmanO;
movefile('Elmanoutput_forecast.mat', 'workfile');
%%ELM仿真
clearvars -except Expert_RE_ace Expert_RE ;
load Input_data data_forecast_anormalization;
load netELM IW B LW TF TYPE;
ybptest=elmpredict(data_forecast_anormalization',IW,B,LW,TF,TYPE)';
ELMO=ybptest;
save ELMoutput_forecast ELMO;
movefile('ELMoutput_forecast.mat', 'workfile');
%%GRNN仿真
clearvars -except Expert_RE_ace Expert_RE ;
load Input_data data_forecast_anormalization;
load netGRNN net;
ybptest=sim(net,data_forecast_anormalization')';
GRNNO=ybptest;
save GRNNoutput_forecast GRNNO;
movefile('GRNNoutput_forecast.mat', 'workfile');

%%组合器决策
clearvars -except Expert_RE_ace Expert_RE ;
load ELmanoutput_forecast;load ELMoutput_forecast;load GRNNoutput_forecast;
load Fcm_data U_Prediction Fcmb_Prediction cluster_n;
load FCM_Combiner_result FCM_result;

%FCM聚类
for cluster_i=1:cluster_n
    
    Expert_FMC_index=find(Fcmb_Prediction==cluster_i)';
    if size(Expert_FMC_index,2)==0
        continue
    else
        
        U_Expert0=U_Prediction(:,Expert_FMC_index);
        [U_Expert_max,~]=max(U_Expert0);
        [U_Expert_min,~]=min(U_Expert0);
        U_Expert_med=median(U_Expert0);
        
        FCM_result_good=FCM_result{cluster_i}(1);%读取训练过程中对专家的评估结果
        FCM_result_med=FCM_result{cluster_i}(2);
        FCM_result_bad=FCM_result{cluster_i}(3);
        U_Expert(FCM_result_good,:)=U_Expert_max;
        if cluster_n==3%聚两类时，舍弃一个最差的专家
            U_Expert(FCM_result_med,:)=U_Expert_med;
            U_Expert(FCM_result_bad,:)=U_Expert_min;
        else
            U_Expert(FCM_result_bad,:)=0;
            U_Expert(FCM_result_med,:)=U_Expert_min;
        end
        
        ElmanO_Expert=ElmanO(Expert_FMC_index,:);
        ELMO_Expert=ELMO(Expert_FMC_index,:);
        GRNNO_Expert=GRNNO(Expert_FMC_index,:);
        Three_Expert_Out=[ElmanO_Expert,ELMO_Expert,GRNNO_Expert];
        
        CM_Expert=Three_Expert_Out*U_Expert;
        CM_Expert=diag(CM_Expert);
        CM(Expert_FMC_index,:)=CM_Expert;
    end
    clear Expert_FMC_index Expert_error U_Expert0 U_Expert_max U_Expert_min U_Expert_med ...
        U_Expert ElmanO_Expert ELMO_Expert GRNNO_Expert Three_Expert_Out CM_Expert;
    
end

load Input_data data_forecast_Depth;
CM_OUT=[data_forecast_Depth,CM];

%%保存为Excel
if exist('委员会机器预测输出.xlsx','file')~=0%%删掉已存在的输出文件
    delete('委员会机器预测输出.xlsx')
end
title={'Depth','CM_'};
xlswrite('委员会机器预测输出.xlsx',title,'CM预测输出');
xlswrite('委员会机器预测输出.xlsx',CM_OUT,'CM预测输出','A2');
%%保存为txt
if exist('委员会机器预测输出.txt','file')~=0%%删掉已存在的输出文件
    delete('委员会机器预测输出.txt')
end
fid=fopen('委员会机器预测输出.txt','a');
fprintf(fid,'%s\n','Depth CM_TOC');
fprintf(fid,'%-7.3f %-5.3f\r\n',CM_OUT');
fclose(fid);