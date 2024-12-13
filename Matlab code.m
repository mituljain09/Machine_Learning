

% Read the Excel file
data = readtable('Pumpkin_Seeds_Dataset.xlsx');


% Extract the categorical column
%'Class' is a categorical column
classColumn = data.Class;

% Convert the cell array of class names to a categorical array
classCategories = categorical(classColumn);

% Logical indexing to encode one class as 0 and the other as 1
encodedClasses = zeros(size(classCategories));  % Initialize with zeros
encodedClasses(classCategories == 'Çerçevelik') = 1;  % Assign 1 to 'Çerçevelik' and 'Ürgüp Sivrisi' as 0
data.Class = encodedClasses;

% Splitting data into X (features) and y (target)
X = data(:, 1:end-1); % Features
y = data(:, end); % Target variable

% Convert 'Class' to array
y = table2array(y);

% Split the data into training and testing sets (70% training, 30% testing)
rng(1); % For reproducibility
cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
idxTrain = training(cv); % Logical index for training set
X_train = X(idxTrain, :); % Training features
y_train = y(idxTrain); % Training labels
X_test = X(~idxTrain, :); % Testing features
y_test = y(~idxTrain); % Testing labels

% Perform feature scaling (standardization) for numerical columns only
varsToScale_train = ~ismember(X_train.Properties.VariableNames, {'Class'}); % Exclude 'Class' column
X_scaled_train = varfun(@(x) (x - mean(x)) ./ std(x), X_train(:, varsToScale_train));

varsToScale_test = ~ismember(X_test.Properties.VariableNames, {'Class'}); % Exclude 'Class' column
X_scaled_test = varfun(@(x) (x - mean(x)) ./ std(x), X_test(:, varsToScale_test));

% Combine scaled features with the 'Class' column
X_scaled_train = [X_scaled_train, X_train(:, ~varsToScale_train)];
X_scaled_test = [X_scaled_test, X_test(:, ~varsToScale_test)];

% Assuming 'data' is your table
correlationMatrix = corr(table2array(X_train));

% Create a heatmap
title('Correlation Heatmap');
customColormap = hot(256);
heatmap(correlationMatrix, 'Colormap', customColormap);
title('Correlation Heatmap');

classCounts = countcats(categorical(data.Class));

uniqueClasses = categories(categorical(data.Class));

% Create a bar plot
figure;
bar(uniqueClasses, classCounts);
xlabel('Class');
ylabel('Count');
title('Count of Classes');


% Perform logistic regression
mdl = fitglm(table2array(X_train), y_train, 'Distribution', 'binomial', 'Link', 'logit');

% Predict using the logistic regression model
y_pred = predict(mdl, table2array(X_test));
threshold = 0.5;  % Define your threshold
y_pred_class = y_pred >= threshold;

% Checking Various Parameters for Logistic Regression
% Reference taken from: https://nl.mathworks.com/matlabcentral/answers/2053757-how-to-evaluate-the-performance-metrics-accuracy-precision-recall-f1score-on-the-tuned-fis?s_tid=srchtitle
confMat = confusionmat(y_test, double(y_pred_class));
accuracy = sum(double(y_pred_class) == y_test) / numel(y_test);
precision = confMat(2,2) / sum(confMat(:,2));
recall = confMat(2,2) / sum(confMat(2,:));
f1Score = 2 * (precision * recall) / (precision + recall);

disp(confMat);
fprintf('Accuracy (Logistic Regression): %.2f%%\n', accuracy * 100);
fprintf('Precision (Logistic Regression): %.2f\n', precision);
fprintf('Recall (Logistic Regression): %.2f\n', recall);
fprintf('F1-score (Logistic Regression): %.2f\n', f1Score);

% Plot ROC curve and calculate AUC for Logistic Regression
[X_ROC_LR, Y_ROC_LR, ~, AUC_LR] = perfcurve(y_test, y_pred, 1);

% Train the Random Forest model
numTrees = 100; % Taking 100 trees
model = TreeBagger(numTrees, table2array(X_train), y_train, 'Method', 'classification');

% Make predictions on the test set
y_pred_RF = predict(model, table2array(X_test));

% Convert the cell array to an array of doubles
y_pred_RF = str2double(y_pred_RF);

% Convert predictions to logical values based on threshold
threshold = 0.5; % Define your threshold
y_pred_class_RF = y_pred_RF >= threshold;

% Checking Various Parameters for Random Forest
confMat_RF = confusionmat(y_test, double(y_pred_class_RF));
accuracy_RF = sum(double(y_pred_class_RF) == y_test) / numel(y_test);
precision_RF = confMat_RF(2,2) / sum(confMat_RF(:,2));
recall_RF = confMat_RF(2,2) / sum(confMat_RF(2,:));
f1Score_RF = 2 * (precision_RF * recall_RF) / (precision_RF + recall_RF);

disp(confMat_RF);
fprintf('Accuracy (Random Forest): %.2f%%\n', accuracy_RF * 100);
fprintf('Precision (Random Forest): %.2f\n', precision_RF);
fprintf('Recall (Random Forest): %.2f\n', recall_RF);
fprintf('F1-score (Random Forest): %.2f\n', f1Score_RF);

% Plot ROC curve for Random Forest and Logistic Regression in the same plot
figure;
plot(X_ROC_LR, Y_ROC_LR, 'b');
hold on;
[X_ROC_RF, Y_ROC_RF, ~, AUC_RF] = perfcurve(y_test, y_pred_RF, 1);
plot(X_ROC_RF, Y_ROC_RF, 'r');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve Comparison');
legend('Logistic Regression', 'Random Forest');
hold off;

% Find and display AUC for Logistic Regression
AUC_LR = trapz(X_ROC_LR, Y_ROC_LR);
fprintf('AUC for Logistic Regression: %.4f\n', AUC_LR);

% Find and display AUC for Random Forest
AUC_RF = trapz(X_ROC_RF, Y_ROC_RF);
fprintf('AUC for Random Forest: %.4f\n', AUC_RF);


% Initialize variables for metrics
avgTrainAUC_LR = 0;
testAUC_LR = 0;
avgTrainError_LR = 0;
testError_LR = 0;
trainTime_LR = 0;
predictTime_LR = 0;

avgTrainAUC_RF = 0;
testAUC_RF = 0;
avgTrainError_RF = 0;
testError_RF = 0;
trainTime_RF = 0;
predictTime_RF = 0;

% Number of folds for cross-validation
numFolds = 10;

% Perform cross-validation
for fold = 1:numFolds
    % Split the data into training and testing sets
    cv = cvpartition(size(X, 1), 'KFold', numFolds);
    idxTrain = training(cv, fold); % Logical index for training set
    X_train = X(idxTrain, :); % Training features
    y_train = y(idxTrain); % Training labels
    X_test = X(~idxTrain, :); % Testing features
    y_test = y(~idxTrain); % Testing labels
    
    % Logistic Regression
    tic;
    mdl_LR = fitglm(table2array(X_train), y_train, 'Distribution', 'binomial', 'Link', 'logit');
    trainTime_LR = trainTime_LR + toc;

    y_pred_LR = predict(mdl_LR, table2array(X_test));
    testAUC_LR = testAUC_LR + computeAUC(y_test, y_pred_LR);
    
    % Random Forest
    tic;
    numTrees = 100; % You can change the number of trees as needed
    model_RF = TreeBagger(numTrees, table2array(X_train), y_train, 'Method', 'classification');
    trainTime_RF = trainTime_RF + toc;

    y_pred_RF = predict(model_RF, table2array(X_test));
    testAUC_RF = testAUC_RF + computeAUC(y_test, str2double(y_pred_RF));
end

% Calculate average metrics over folds
avgTrainAUC_LR = avgTrainAUC_LR / numFolds;
testAUC_LR = testAUC_LR / numFolds;
avgTrainError_LR = 1 - avgTrainAUC_LR;
testError_LR = 1 - testAUC_LR;
trainTime_LR = trainTime_LR / numFolds;

avgTrainAUC_RF = avgTrainAUC_RF / numFolds;
testAUC_RF = testAUC_RF / numFolds;
avgTrainError_RF = 1 - avgTrainAUC_RF;
testError_RF = 1 - testAUC_RF;
trainTime_RF = trainTime_RF / numFolds;

% Display the results
disp('Logistic Regression Metrics:');
disp(['Avg Train AUC: ' num2str(avgTrainAUC_LR)]);
disp(['Test AUC: ' num2str(testAUC_LR)]);
disp(['Avg Train Error: ' num2str(avgTrainError_LR)]);
disp(['Test Error: ' num2str(testError_LR)]);
disp(['Train Time: ' num2str(trainTime_LR) ' seconds']);

disp('Random Forest Metrics:');
disp(['Avg Train AUC: ' num2str(avgTrainAUC_RF)]);
disp(['Test AUC: ' num2str(testAUC_RF)]);
disp(['Avg Train Error: ' num2str(avgTrainError_RF)]);
disp(['Test Error: ' num2str(testError_RF)]);
disp(['Train Time: ' num2str(trainTime_RF) ' seconds']);

% Helper function to compute AUC
function auc = computeAUC(y_true, y_pred)
    [~, ~, ~, auc] = perfcurve(y_true, y_pred, 1);
end

