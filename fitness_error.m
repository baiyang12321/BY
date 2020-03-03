function error = fitness_error(X, target_train, expert1, expert2, expert3)
  %用于遗传算法的目标函数，计算训练集总误差

Overall_out=expert1.*X(1)+expert2.*X(2)+expert3.*X(3);
error=mean(abs(Overall_out-target_train)./target_train);

end