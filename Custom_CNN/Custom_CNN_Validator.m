clear all;
close all;
clc;

%%% Get the Data.
[train_data,train_labels,~,~] = helperCIFAR10Data.load('./data');
[height,width,channels,num_samples] = size(train_data);

disp('Done loading the Training Dataset');

%%% Formation of the Training and the validation dataset for cross
%%% validation.
X_train = train_data(:,:,:,1:46000);
Y_train = train_labels(1:46000);

X_validation = train_data(:,:,:,46001:num_samples);
Y_validation = train_labels(46001:num_samples);
disp('Done forming the Validation Dataset:');

load('Pretrained_Networks\trained_cnn_validation.mat');

output_validation_predicted = classify(net,X_validation);
accuracy_validation = sum(output_validation_predicted == Y_validation)/numel(Y_validation);
disp('The Accuracy on the Validation Dataset is:');
disp(accuracy_validation);


stats = confusionmatStats(Y_validation,output_validation_predicted);
save('./data/results/stats_validation.mat','stats');
disp('Done Saving the Performance Parameters for each class on Validation Dataset.');