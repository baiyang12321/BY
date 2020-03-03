clc;clear;
%%������FCM�㷨����Ȩ��ѡ��
%%FCM����ʱ������ѵ�����ݺ�Ԥ�����ݡ�
%%���С�������ݣ�������һ������֤������ѵ��
clc;clear;warning off;
%ɾ�������ļ���
if exist('workfile','dir')~=0
    rmdir('workfile', 's');
end
mkdir('workfile');%%����һ���ļ���
addpath('workfile');%%�����ļ�����ӵ�·��
try
    [adata_train,bdata_train,cdata_train]=xlsread('train.xlsx');
    [adata_forecast,bdata_forecast,cdata_forecast]=xlsread('prediction.xlsx');
catch
    errordlg('���ļ�ʧ��','Wrong File');
    return
end

data_train=adata_train;
P_column=str2num('2 3 4 5 6 8');%ѵ����������2 3 4 5 6 7 8 9 10
T_column=str2double('9');%��ǩ
data_forecast=adata_forecast(:,P_column);
data_forecast_Depth=adata_forecast(:,1);

dataX=data_train(:,P_column);
dataY=data_train(:,T_column);
dataX(isnan(dataX))=0;
dataY(isnan(dataY))=0;%��ֵȫ��Ϊ��

if exist('ίԱ������������ݼ�.xlsx','file')~=0%%ɾ���Ѵ��ڵ�����ļ�
    delete('ίԱ������������ݼ�.xlsx')
end
xlswrite('ίԱ������������ݼ�.xlsx',dataX,'inputdata');
xlswrite('ίԱ������������ݼ�.xlsx',dataY,'outputdata');

%ѵ�����ݹ�һ��
data_total=[dataX;data_forecast];
[dataX_total_anormalization ,inputps]=mapminmax(data_total');%��һ��Ĭ��Ϊ-1~1
dataX_total_anormalization=dataX_total_anormalization';
data_train_anormalization=dataX_total_anormalization(1:size(dataX,1),:);%ѵ�����ݹ�һ�����
data_forecast_anormalization=dataX_total_anormalization(size(dataX,1)+1:end,:);%Ԥ�����ݹ�һ�����
%K�۽�����֤
K_fold=1;%����
% K_fold=size(dataY,1);%��һ������֤
indices =crossvalind('Kfold', size(dataX,1), K_fold);
train_P_data=cell(1,K_fold);%%Ԥ��������������ڴ�
train_T_data=cell(1,K_fold);%%Ԥ��������������ڴ�
test_P_data=cell(1,K_fold);%%Ԥ��������������ڴ�
test_T_data=cell(1,K_fold);%%Ԥ��������������ڴ�
for i = 1:K_fold %ѭ����Σ��ֱ�ȡ����i������Ϊ����������������������Ϊѵ������
    cross_num_test = (indices == i);%���һ����֤���������Ϊѵ����
    cross_num_train = ~cross_num_test;
    train_P_data{i} = data_train_anormalization(cross_num_train, :);%��ȡ��ѵ����
    train_T_data{i} = dataY(cross_num_train, :);
    test_P_data{i} = data_train_anormalization(cross_num_test, :);%��ȡ����֤��
    test_T_data{i} = dataY(cross_num_test, :);
end

[dataX_m,~]=size(dataX);
%%FCM-cluster
cluster_n=2;%�����
options = [2;500;1e-5;0];
data=[dataX,dataY];

[center_total, U_total, obj_fcn_total] = fcm(dataX_total_anormalization, cluster_n, options);
U_Core=U_total(:,1:dataX_m);
U_Prediction=U_total(:,dataX_m+1:end);

[Fcma_Core,Fcmb_Core]=max(U_Core);%%��������ʵ�ʷ���
Fcmb_Core=Fcmb_Core';
[Fcma_Prediction,Fcmb_Prediction]=max(U_Prediction);%%Ԥ������ʵ�ʷ���
Fcmb_Prediction=Fcmb_Prediction';

%FCMԤ�⼯���������ݹ�ϵ
FCM_figure(dataX_total_anormalization,Fcmb_Core,Fcmb_Prediction);

save Input_data data_train adata_forecast data_forecast_Depth data_train_anormalization data_forecast_anormalization ...
    train_P_data train_T_data test_P_data test_T_data dataY inputps;
save Fcm_data Fcma_Core Fcmb_Core Fcma_Prediction Fcmb_Prediction K_fold U_Core U_Prediction cluster_n;
movefile('Input_data.mat', 'workfile');
movefile('Fcm_data.mat', 'workfile');
%% Elman������
clear;
%%��������
load Input_data train_P_data train_T_data test_P_data test_T_data data_train_anormalization dataY;
load Fcm_data K_fold;
%%����Ľ�����ѵ��
% ����ѭ�������ò�ͬ�����ز���Ԫ����
nn=10:2:30;
Elman_cross_RE=zeros(length(nn),K_fold);
for k=1:K_fold
    Re_train=zeros(length(nn),1);
    Re_test=zeros(length(nn),1);
    for i=1:length(nn)
        % ����Elman������ ���ز�Ϊnn(i)����Ԫ
        
        net=newelm(minmax(train_P_data{k}'),[nn(i),1],{'tansig','purelin'},'traingdm');
        % ��������ѵ������
        net.trainparam.epochs = 200;
        net.trainParam.lr = 0.01;
        net.trainParam.mc = 0.9;
        net.trainParam.goal = 1e-2;
        net.trainparam.show = 50;
        net.trainParam.showWindow = false; % ����ʾѵ������
        % Elman����ѵ��
        net=train(net,train_P_data{k}',train_T_data{k}','useGPU','no');
        % Ԥ������
        ty_train=sim(net,train_P_data{k}')';
        ty_test=sim(net,test_P_data{k}')';
        % �������
        %     test_out=mapminmax('reverse',ty,outputps);
        Re_train(i,:)=mean(abs(ty_train-train_T_data{k})./train_T_data{k});
        Re_test(i,:)=mean(abs(ty_test-test_T_data{k})./test_T_data{k});
        % ��ʼ������
        net=init(net);
    end
    Elman_cross_RE(:,k)=Re_train+Re_test;%��¼ÿK��ѵ������֤����
end
Elman_cross_RE_sum=sum(Elman_cross_RE,2);
[~,Elman_cross_parameter]=min(Elman_cross_RE_sum);
Elman_hidnumber=nn(Elman_cross_parameter);%�õ�������С�Ĳ���

% ��ʼ������
init(net);
%%��ʽ��������
net=newelm(minmax(data_train_anormalization'),[Elman_hidnumber,1],{'tansig','purelin'},'traingdm');
% ��������ѵ������
net.trainparam.epochs = 2000;%����������ѵ��������
net.trainParam.lr = 0.001;%ѧϰ��
net.trainParam.mc = 0.9;%���Ӷ�������
net.trainParam.goal = 1e-3;%ѵ��Ŀ����С���
net.trainparam.show = 20;%��ʵƵ�ʣ���������Ϊÿѵ��20����ʾһ��
net.trainParam.showWindow = false; % ����ʾѵ������
net.trainparam.min_fail=5;% ���ȷ��ʧ�ܴ���
net.trainparam.min_grad=1e-6;% ��С�ݶ�

% Elman����ѵ��
[net,tr,Y,E]=train(net,data_train_anormalization',dataY','useGPU','no');
% Ԥ������
ybptest=sim(net,data_train_anormalization')';
% �������
ElmanO=ybptest;
Elman_RE=abs(ElmanO-dataY)./dataY;
Elman_ave=mean(Elman_RE);
Expert_RE=Elman_RE;%%ÿ���������������
Expert_RE_ace=Elman_ave;%%���в���������ƽ�����

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save netElman net Elman_hidnumber ElmanO;
movefile('netElman.mat', 'workfile');

%% ����ѧϰ���ع����
clearvars -except Expert_RE_ace Expert_RE ;
%%��������
load Input_data train_P_data train_T_data test_P_data test_T_data data_train_anormalization dataY;
load Fcm_data K_fold;

% ����ѭ�������ò�ͬ�Ĳ���
nn=2:2:40;
ELM_cross_RE=zeros(length(nn),K_fold);
for k=1:K_fold
    for i=1:length(nn)
        
        [IW,B,LW,TF,TYPE] = elmtrain(train_P_data{k}',train_T_data{k}',nn(i),'sig',0);
        %%ELM�������
        ty_train=elmpredict(train_P_data{k}',IW,B,LW,TF,TYPE)';
        ty_test=elmpredict(test_P_data{k}',IW,B,LW,TF,TYPE)';
        
        % �������
        Re_train(i,:)=mean(abs(ty_train-train_T_data{k})./train_T_data{k});
        Re_test(i,:)=mean(abs(ty_test-test_T_data{k})./test_T_data{k});
        clear IW B LW TF TYPE;
        
    end
    ELM_cross_RE(:,k)=Re_train+Re_test;%��¼ÿK��ѵ������֤����
end
ELM_cross_RE_sum=sum(ELM_cross_RE,2);
[~,ELM_cross_parameter]=min(ELM_cross_RE_sum);
ELM_M=nn(ELM_cross_parameter);%�õ�������С�Ĳ���

% ��ʼ������
clear IW B LW TF TYPE;
%%��ʽ��������ѧϰ��
[IW,B,LW,TF,TYPE] = elmtrain(data_train_anormalization',dataY',ELM_M,'sig',0);

%%ELM�������
ybptest=elmpredict(data_train_anormalization',IW,B,LW,TF,TYPE)';

% �������
ELMO=ybptest;
ELM_RE=abs(ELMO-dataY)./dataY;
ELM_ave=mean(ELM_RE);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save netELM IW B LW TF TYPE ELM_M ELMO;
movefile('netELM.mat', 'workfile');

Expert_RE=[Expert_RE,ELM_RE];%%ÿ���������������
Expert_RE_ace=[Expert_RE_ace;ELM_ave];%%���в���������ƽ�����

%% GRNN������
clearvars -except Expert_RE_ace Expert_RE ;
%%��������
load Input_data train_P_data train_T_data test_P_data test_T_data data_train_anormalization dataY;
load Fcm_data K_fold;

% ����ѭ����Ѱ����Ѳ���
nn=0.3:0.1:0.9;
GRNN_cross_RE=zeros(length(nn),K_fold);
for k=1:K_fold
    for i=1:length(nn)
        % ����Elman������ �����ٶ�Ϊnn(i)
        net=newgrnn(train_P_data{k}',train_T_data{k}',nn(i));
        % Ԥ������
        ty_train=sim(net,train_P_data{k}')';
        ty_test=sim(net,test_P_data{k}')';
        % �������
        Re_train(i,:)=mean(abs(ty_train-train_T_data{k})./train_T_data{k});
        Re_test(i,:)=mean(abs(ty_test-test_T_data{k})./test_T_data{k});
        % ��ʼ������
        net=init(net);
    end
    GRNN_cross_RE(:,k)=Re_train+Re_test;%��¼ÿK��ѵ������֤����
end
GRNN_cross_RE_sum=sum(GRNN_cross_RE,2);
[~,GRNN_cross_parameter]=min(GRNN_cross_RE_sum);
GRNN_Spread=nn(GRNN_cross_parameter);%�õ�������С�Ĳ���

% ��ʼ������
init(net);
%%��ʽ��������GRNN����
net=newgrnn(data_train_anormalization',dataY',GRNN_Spread);
% Ԥ������
ybptest=sim(net,data_train_anormalization')';

% �������
GRNNO=ybptest;
GRNN_RE=abs(GRNNO-dataY)./dataY;
GRNN_ave=mean(GRNN_RE);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save netGRNN net GRNN_Spread GRNNO;
movefile('netGRNN.mat', 'workfile');

Expert_RE=[Expert_RE,GRNN_RE];%%ÿ���������������
Expert_RE_ace=[Expert_RE_ace;GRNN_ave];%%���в���������ƽ�����

%% ίԱ����������Train
clearvars -except Expert_RE_ace Expert_RE ;
%%��ȡ����
load netElman ElmanO;load netELM ELMO;load netGRNN GRNNO;load Input_data dataY;
load Fcm_data Fcmb_Core U_Core cluster_n;

%Ѱ����õ�ר��
for cluster_i=1:cluster_n
    
    Expert_FMC_index=find(Fcmb_Core==cluster_i)';
    if size(Expert_FMC_index,2)==0
        continue
    else
        Expert_error=Expert_RE(Expert_FMC_index,:);
        Expert_error_average=mean(Expert_error,1);
        [~,FCM_result_good]=min(Expert_error_average);%%��õ�ר��
        [~,FCM_result_bad]=max(Expert_error_average);%%���ר��
        FCM_result_med=median(Expert_error_average,2);%%���õ�ר��
        FCM_result_med=find(Expert_error_average==FCM_result_med);
        
        U_Expert0=U_Core(:,Expert_FMC_index);
        [U_Expert_max,~]=max(U_Expert0);
        [U_Expert_min,~]=min(U_Expert0);
        U_Expert_med=median(U_Expert0);
        U_Expert(FCM_result_good,:)=U_Expert_max;
        
        if cluster_n==3%������ʱ������һ������ר��(Ŀǰֻ֧�����������)
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
disp('��ר��������:')
disp(Expert_RE_ace)
disp('ίԱ�����������:')
disp(CM_ave)
%%%%%%%%%@%%%%%%%%%@%%%
%%%%%%%%@%%%%%%%%%@%%%%
%%%%%%%@%%%%%%%%%@%%%%%
%%%%%%@%%%%%%%%%@%%%%%%
%%%%%@%%%%%%%%%@%%%%%%%
%%%%@%%%%%%%%%@%%%%%%%%
%%%@%%%%%%%%%@%%%%%%%%%
%% forcast
%%Elman����
clearvars -except Expert_RE_ace Expert_RE ;
load Input_data data_forecast_anormalization;
load netElman net;
ybptest=sim(net,data_forecast_anormalization')';
ElmanO=ybptest;
save Elmanoutput_forecast ElmanO;
movefile('Elmanoutput_forecast.mat', 'workfile');
%%ELM����
clearvars -except Expert_RE_ace Expert_RE ;
load Input_data data_forecast_anormalization;
load netELM IW B LW TF TYPE;
ybptest=elmpredict(data_forecast_anormalization',IW,B,LW,TF,TYPE)';
ELMO=ybptest;
save ELMoutput_forecast ELMO;
movefile('ELMoutput_forecast.mat', 'workfile');
%%GRNN����
clearvars -except Expert_RE_ace Expert_RE ;
load Input_data data_forecast_anormalization;
load netGRNN net;
ybptest=sim(net,data_forecast_anormalization')';
GRNNO=ybptest;
save GRNNoutput_forecast GRNNO;
movefile('GRNNoutput_forecast.mat', 'workfile');

%%���������
clearvars -except Expert_RE_ace Expert_RE ;
load ELmanoutput_forecast;load ELMoutput_forecast;load GRNNoutput_forecast;
load Fcm_data U_Prediction Fcmb_Prediction cluster_n;
load FCM_Combiner_result FCM_result;

%FCM����
for cluster_i=1:cluster_n
    
    Expert_FMC_index=find(Fcmb_Prediction==cluster_i)';
    if size(Expert_FMC_index,2)==0
        continue
    else
        
        U_Expert0=U_Prediction(:,Expert_FMC_index);
        [U_Expert_max,~]=max(U_Expert0);
        [U_Expert_min,~]=min(U_Expert0);
        U_Expert_med=median(U_Expert0);
        
        FCM_result_good=FCM_result{cluster_i}(1);%��ȡѵ�������ж�ר�ҵ��������
        FCM_result_med=FCM_result{cluster_i}(2);
        FCM_result_bad=FCM_result{cluster_i}(3);
        U_Expert(FCM_result_good,:)=U_Expert_max;
        if cluster_n==3%������ʱ������һ������ר��
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

%%����ΪExcel
if exist('ίԱ�����Ԥ�����.xlsx','file')~=0%%ɾ���Ѵ��ڵ�����ļ�
    delete('ίԱ�����Ԥ�����.xlsx')
end
title={'Depth','CM_'};
xlswrite('ίԱ�����Ԥ�����.xlsx',title,'CMԤ�����');
xlswrite('ίԱ�����Ԥ�����.xlsx',CM_OUT,'CMԤ�����','A2');
%%����Ϊtxt
if exist('ίԱ�����Ԥ�����.txt','file')~=0%%ɾ���Ѵ��ڵ�����ļ�
    delete('ίԱ�����Ԥ�����.txt')
end
fid=fopen('ίԱ�����Ԥ�����.txt','a');
fprintf(fid,'%s\n','Depth CM_TOC');
fprintf(fid,'%-7.3f %-5.3f\r\n',CM_OUT');
fclose(fid);