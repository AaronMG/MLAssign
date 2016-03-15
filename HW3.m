function HW3()
clc;
clear;
% Needs to be changed if you'd like to run the code. 
loc = 'E:\gonza647\Courses\CSE847 - Machine Learning\HW3\diabetes.mat';
load(loc);

bias = ones(200, 1);
x_test = [bias x_test];
bias = ones(242, 1);
x_train = [bias x_train];
factor = 240/5;

count = 1;
lambdas = [-5 -4 -3 -2 -1 0 1];
for l = 1:7
   lambdas(l) = 10^lambdas(l);
end



%Part 2, train with the training sets, and test against both training and
%test sets. Do this for every lambda
for l = 1: 7
   trainErr(l) = ridgeRegression(x_train, y_train, x_train, y_train, lambdas(l)); 
   testErr(l) = ridgeRegression(x_train, y_train, x_test, y_test, lambdas(l));
   sprintf('Lambda = %d\tTrainErr = %d\tTestErr = %d\n', lambdas(l), trainErr(l), testErr(l))
end
figure
semilogx(lambdas, trainErr,'--') 
hold on
semilogx(lambdas, testErr) 
hold on;
plot(0.1, testErr(5), 'g*', 0.1, trainErr(5), 'g*');
legend('Training Err', 'Testing Err', 'Optimal Lambda');
%Part 3 Begin cross validation
for power = 1: 7    
    for i = 1:5        
        % Partition our training set into 5 equal pieces
        start = factor*(i-1)+1;
        fin = factor*i;
        if i == 5 fin = 242; end
        test = x_train([start:fin],:);
        truth = y_train([start:fin],:);
        xtrain = x_train;
        xtrain([start:fin],:) = [];
        ytrain = y_train;
        ytrain([start:fin],:) = [];
        CVErr(i) = ridgeRegression(xtrain, ytrain, test, truth, lambdas(power));
    end
    % Take the mean of the lambdas CVErrors
    meanerr = sum(CVErr)/5;
    % Store all 7 mean CVErrors to find the optimal lambda later
    lambdaErrs(count) = meanerr;
    count = count + 1;
end

[M, I] = min(lambdaErrs);
% Taking the optimal lambda, run on the test set. 
finErr = ridgeRegression(x_train, y_train, x_test, y_test, lambdas(I));
sprintf('Training our model using Cross Validation, we compute an optimal lambda of %0.2d,\n and an error of %d on the test set.\n', lambdas(I), finErr)
end

%Part 1
function out = ridgeRegression(xtrain, ytrain, test, truth, lambda)
% Calculate the weights
sqrd = xtrain.'*xtrain;
[n,m] = size(sqrd);
iden = eye(n,m);
weights = inv(iden + lambda*inv(sqrd)) * inv(sqrd)*xtrain.'*ytrain;
% With the weights we can now calculate a prediction
pred = test * weights;
[r,c] = size(test);
samples = r;
% Store all 5 CVErrors
out = (1/samples) * sum((truth - pred).^2);
end