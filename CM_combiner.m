function weights = CM_combiner(expert1, expert2, expert3, target_train)
% ���ڻع�Ԥ���ίԱ�������ѵ�������õ��Ƕ�Ӧ3������ϵͳ��Ȩ��ֵ

Aeq = [1 1 1];
Beq = 1;

LB = [0 0 0];   %Ȩ������

%ʹ���Ŵ��㷨�õ����ŵ�Ȩ�ط��䣬��ʹ��ίԱ������������ȷ�����
options = gaoptimset();
options.Generations = 100;
options.Display = 'iter';
options.PloyFcns = @gaplotbestf;
weights = ga( @(X)fitness_error(X, target_train, expert1, expert2, expert3), 3, [],[], Aeq, Beq, LB, [],[], options);
% weights = ga( @(X)mse_fitfunc(X, target_train, BPNN_target, SVM_target, FIS_target), 3, [],[], Aeq, Beq, LB, [],[], options);

end