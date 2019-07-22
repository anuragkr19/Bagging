function [maxValue] = predictMax(X_test,theta_values,m)
    %method to perform voting on the output label predicted by classifier
    %and returning the maximum voted lable as output
    
    countOne = 0;
    countZero = 0;
    
    for p = 1:m
        y_pred = predict(sigmoid(X_test * theta_values(:,p)));
        if y_pred == 1
            countOne = countOne + 1;
        else
            countZero = countZero + 1;
        end            
    end
    
    if countOne > countZero
        maxValue = 1;
    else
        maxValue = 0;
    end
end