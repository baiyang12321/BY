function weights = CM_combiner(expert1, expert2, expert3, target_train)
% 用于回归预测的委员会机器的训练，最后得到是对应3个智能系统的权重值

Aeq = [1 1 1];
Beq = 1;

LB = [0 0 0];   %权重下限

%使用遗传算法得到最优的权重分配，即使得委员会机器输出的正确率最高
options = gaoptimset();
options.Generations = 100;
options.Display = 'iter';
options.PloyFcns = @gaplotbestf;
weights = ga( @(X)fitness_error(X, target_train, expert1, expert2, expert3), 3, [],[], Aeq, Beq, LB, [],[], options);
% weights = ga( @(X)mse_fitfunc(X, target_train, BPNN_target, SVM_target, FIS_target), 3, [],[], Aeq, Beq, LB, [],[], options);

end