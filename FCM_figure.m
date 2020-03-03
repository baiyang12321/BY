function FCM_figure(dataX_total_anormalization,Fcmb_Core,Fcmb_Prediction)
%%���ɷַ���
[~,score,~,~,~,~] = pca(dataX_total_anormalization);
score_Core=score(1:size(Fcmb_Core,1),:);
score_prediction=score(size(Fcmb_Core,1)+1:end,:);
%coeff���ɷֱ��ʽ��ϵ������,������������
%score���ɷֵ÷�����,���µ�����ϵ�»�õ�����
%latent�������ϵ�����������ֵ����,�Ӵ�С����
%tsquare���ݻ�����T2ͳ����Ѱ�Ҽ�������
%explained������
%mu���еľ�ֵ
figure(1);
numClust=max(Fcmb_Prediction);
color1={'m','y','c'};
color2={'r','g','b'};
for i=1:numClust

[pca_m,pca_n]=find(Fcmb_Prediction==i);
score_cluster=score_prediction(pca_m,:);
scatter(score_cluster(:,1),score_cluster(:,2),25,color1{i});
legend_str{i}=['cluster Forecast' num2str(i)];
hold on;
end


hold on;

for i=1:numClust

[pca_m,pca_n]=find(Fcmb_Core==i);
score_cluster=score_Core(pca_m,:);
scatter(score_cluster(:,1),score_cluster(:,2),25,color2{i},'o','filled');
legend_str{i+numClust}=['cluster Core' num2str(i)];
h1=legend(legend_str,'Location','NorthEast');
hold on;
% title('results')
end
h1=legend(legend_str,'Location','NorthWest');

%%��ȥ�Ϸ����ҷ��Ŀ̶�
box off;
ax2 = axes('Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on;
end