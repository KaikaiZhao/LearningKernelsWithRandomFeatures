%% Learning kernels with Random Features, example script

% Generate some data
% Create normally distributed points and let the true classifier between
% classes be a specified radius
clear all;
close all;
clc;
rng(7117);
path = '/home/zkk/Downloads/learning-kernels';% change the path to yours
addpath(genpath(path));
for d=2:15
n = 10000;disp(['The current d is ',num2str(d)]);
%d = 2;
Xtrain = randn(d, n);
Xtest = randn(d, n/10);

ytrain = sqrt(sum(Xtrain.*Xtrain,1))>sqrt(d);
ytrain = ytrain'*2-1;
ytest = sqrt(sum(Xtest.*Xtest,1))>sqrt(d);
ytest = ytest'*2-1;

%% Optimize kernel
% This example uses the Gaussian kernel and chi-square divergence
Nw = 2e4;
rho = Nw*0.005;
tol = 1e-11;
[Wopt, bopt, alpha, alpha_distrib] = optimizeGaussianKernel(Xtrain, ytrain, Nw, rho, tol);
disp(['The number of optimized features is ',num2str(length(alpha))]);
%%
% only for d=2, setup colors for positive(blue) and negative(red) datapoints
if d==2
    for i=1:length(ytrain)
        if ytrain(i) > 0
            c(i,:)=[0 0 1];
        else
            c(i,:)=[1 0 0];
        end
    end
    figure(1); % for fig1(a) in Aman's paper
    scatter(Xtrain(1,:),Xtrain(2,:),30,c);hold on;
    scatter(Wopt(1,:),Wopt(2,:),30,'y');hold off;
    figure(2); % for offset v_m which is not shown in Aman's paper
    plot(bopt,'Marker','o','Color',[0 0 1],'MarkerFaceColor','red','LineStyle','none');
    % Take a look at what the distirbution looks like
    figure(3);
    plot(sort(alpha));
    xlabel('Feature');
    ylabel('Probability');
end
%% Create random features using optimized kernel
% pick a number of random features to use for the model
D = length(alpha);%250;
% generate parameters for the optimized kernel
[D_opt, W_opt, b_opt] = createOptimizedGaussianKernelParams(D, Wopt, bopt, alpha_distrib);
% create optimized features using the training data and test data
Z_opt_train = createRandomFourierFeatures(D_opt, W_opt, b_opt, Xtrain);
Z_opt_test = createRandomFourierFeatures(D_opt, W_opt, b_opt, Xtest);

% Generate regular Gaussian features for comparison
W = randn(d,D);
b = rand(1,D)*2*pi;
Z_train = createRandomFourierFeatures(D, W, b, Xtrain);
Z_test = createRandomFourierFeatures(D, W, b, Xtest);

%% Train models
% For simplicity, train linear regression models (even though this is a
% classification problem!)
meany = mean(ytrain);
lambda = .05;
%w_opt = (Z_opt_train * Z_opt_train' + lambda * eye(D_opt)) \ (Z_opt_train * (ytrain-meany));
%w = (Z_train * Z_train' + lambda * eye(D)) \ (Z_train * (ytrain-meany));
% Note that we don't bother scaling the features by sqrt(alpha) since we
% can absorb that factor into w_opt for this ridge regression model

% If you have the ability to use smarter models, then you can try:
%mdl_GK = fitglm(Z_train', (ytrain+1)/2,'linear', 'Distribution', 'binomial','link','logit');
%mdl_OK = fitglm(Z_opt_train', (ytrain+1)/2, 'linear', 'Distribution', 'binomial','link','logit');
mdl_GK = fitglm(Z_train', (ytrain+1)/2,'Distribution', 'binomial');
mdl_OK = fitglm(Z_opt_train', (ytrain+1)/2,'Distribution', 'binomial');
Ytrain_GK = predict(mdl_GK,Z_train');
Ytrain_OK = predict(mdl_OK,Z_opt_train');
Ytest_GK = predict(mdl_GK,Z_test');
Ytest_OK = predict(mdl_OK,Z_opt_test');
error_train_GK(d-1) = mean((round(Ytrain_GK)*2-1)~=ytrain);
error_test_GK(d-1) = mean((round(Ytest_GK)*2-1)~=ytest);
error_train_OK(d-1) = mean((round(Ytrain_OK)*2-1)~=ytrain);
error_test_OK(d-1) = mean((round(Ytest_OK)*2-1)~=ytest);
end
figure(4); % for fig1(b) in Aman's paper
plot(2:15,error_train_GK,'b','LineWidth',2);hold on;
plot(2:15,error_test_GK,'b--','LineWidth',2);
plot(2:15,error_train_OK,'r','LineWidth',2);
plot(2:15,error_test_OK,'r--','LineWidth',2);
legend('GK-train','GK-test','OK-train','OK-test');hold off;
% or
% mdl = fitcsvm(Z_train', ytrain, 'KernelFunction', 'linear', 'ClassNames', [-1, 1]);
% and then change the error computation code accordingly for the logistic
% regression or SVM models respectively.

%% errors
%calculate errors on training set
disp(['Fraction of positives (train): ' num2str(sum(ytrain==1)/length(ytrain))])
[err,fp, fn] = computeError(Z_train, w, meany, ytrain);
disp(['Regular train error: ' num2str(err)]);
disp(['false positives: ' num2str(fp)]);
disp(['false negatives: ' num2str(fn)]);
disp(' ')
[err,fp, fn] = computeError(Z_opt_train, w_opt, meany, ytrain);
disp(['Optimized train error: ' num2str(err)])
disp(['false positives: ' num2str(fp)])
disp(['false negatives: ' num2str(fn)])
disp(' ')
disp(' ')

disp(['Fraction of positives (test): ' num2str(sum(ytest==1)/length(ytest))])
[err,fp, fn] = computeError(Z_test, w, meany, ytest);
disp(['Regular test error: ' num2str(err)]);
disp(['false positives: ' num2str(fp)]);
disp(['false negatives: ' num2str(fn)]);
disp(' ')
[err,fp, fn] = computeError(Z_opt_test, w_opt, meany, ytest);
disp(['Optimized test error: ' num2str(err)])
disp(['false positives: ' num2str(fp)])
disp(['false negatives: ' num2str(fn)])