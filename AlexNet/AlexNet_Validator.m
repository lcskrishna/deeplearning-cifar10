clear all;
close all;
clc;

%% Loading of the Training Dataset and splitting into valdiation dataset.
[train_data,train_labels,~,~] = helperCIFAR10Data.load('data/');
[height,width,channels,num_samples] = size(train_data);

%% Formation of the Training and the validation dataset for cross validation.
X_train = train_data(:,:,:,1:46000);
Y_train = train_labels(1:46000);

X_validation = train_data(:,:,:,46001:num_samples);
Y_validation = train_labels(46001:num_samples);

disp('Done loading the Validation Dataset');

disp('Loading the trained model..:');
load('data\networks\trained_alexnet_validation.mat');
disp('Done with loading the trained model');

output_predicted = classify(net,X_validation);
accuracy_validation = sum(output_predicted == Y_validation)/numel(Y_validation);
disp('Accuracy on the Validation Dataset is:');
disp(accuracy_validation);

stats = confusionmatStats(Y_validation,output_predicted);
save('data/results/stats_alexnet_validation.mat','stats');
