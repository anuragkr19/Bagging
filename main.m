%external reference used
% https://www.coursera.org/learn/machine-learning/home/week/3
% https://www.coursera.org/learn/machine-learning/programming/ixFof/logistic-regression
% http://ciml.info/dl/v0_9/ciml-v0_9-ch11.pdf
% www.biostat.wisc.edu/~dpage/cs760/

%Main entry of the program

function   main()

% Fetching  data from test and train files

data = load('train.txt');
testData = load('test.txt');
X = data(:, 1:4);
X_test = testData(:,1:4);
y_test = testData(:,5);
y = data(:, 5);
m = length(X);

%end loading  data

% getting user input


%preprocessing data

% Preparing data matrix using Training dataset
[m, n] = size(X);

% Add intercept term to training datasets X
X = [ones(m, 1) X];

% Add intercept term to test set X_test
X_test = [ones(length(X_test), 1) X_test];

%setting learning rate and no of iterations for individual logistic
%regression classifier
alpha = 0.01;
noofiterations = 1200;
initial_theta = randi([0, 1],n+1,1);
%matrix to store theta values per iterations
theta_values = zeros(length(initial_theta),m);

%performing classification using single logistic regression

[theta_single_classifier] = gradientDescent(X, y, initial_theta, alpha, noofiterations);
result_test = 0;
accuracy_test = 0;
error_count = 0;

%  Predict output label for a test data set using single logistic
%  regression classifier
for k = 1:length(X_test)
    prob = sigmoid(X_test(k,:) * theta_single_classifier);
    if prob >=0.5
        result_test = 1;
    else
        result_test = 0;
    end
    if result_test == y_test(k)
        accuracy_test = accuracy_test + 1;
    end
    fprintf('predicted class label is %d and true class label was %d\n',result_test,y_test(k));
end

error_count = length(X_test)- accuracy_test;
fprintf('Accuracy and error rate for single logistic regression classifier on test data %d%%,%d%%\n',round((error_count/length(y_test))*100),round((accuracy_test/length(y_test))*100));


%end of single logistic regression

%end preprocessing data

order = input("Enter number of iterations for bagging routine.Please enter value from this set [10,50,100]\n");
while order == 10 || order == 50 || order == 100
   for j =1:order
   %Cretaing m many bootstrap training dataset. using random sampling of data with replacement
   X1 = datasample(X,size(X,1)); 
   %random initialization of initial parameter values
   initial_theta =  randi([0, 1],n+1,1);
   %getting learned parameter
   [theta] = gradientDescent(X1, y, initial_theta, alpha, noofiterations);
   %storing theta values in the matrix
   theta_values(:,j) = theta;
   end

   %counting no of correct labels
   correct_result = 0;

   %Predicting label of test dataset 
   for k = 1:length(X_test)
       %getting output label based on voting performed on the result of all
       %classifiers
       [maxValue] = predictMax(X_test(k,:),theta_values,order);
       if maxValue == y_test(k)
            correct_result = correct_result + 1;
       end
       fprintf('For a test data  predicted class label is %d and true class label was %d\n',maxValue,y_test(k));
   end

   %printing accuracy and error rate 
   p = length(X_test);
   result = correct_result/p;
   error_rate = (p - correct_result)/p;
   
   fprintf('Accuracy and Error rate for ensemble classifier after iteration %d is %d%%,%d%%\n',order,round(result*100),round(error_rate*100));
   
   response = input('Please enter your response to run this programme again.For Yes enter 1 and for No enter 0 ?');
   if response == 1
       order = input("Enter number of iterations for bagging routine.Please enter value from this set [10,50,100]\n");
       continue
   else
       break
   end
end


%end of programme

end

