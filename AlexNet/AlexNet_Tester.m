clear all;
close all;
clc;

%% Loading of the Training Dataset and splitting into valdiation dataset.
[~,~,test_data,test_labels] = helperCIFAR10Data.load('data/');
disp('Done loading the Testing Dataset');
%% Loading the Trained Model for CIFAR 10.
disp('Loading the trained model..:');
load('data\networks\trained_alexnet_validation.mat');
disp('Done with loading the trained model');


%% Calculate the Accuracy and Performance Measures.
output_test_predicted = classify(net,test_data);
accuracy_testing= sum(output_test_predicted==test_labels)/numel(test_labels);
disp('Accuracy on the Testing Dataset is:');
disp(accuracy_testing);

stats = confusionmatStats(test_labels,output_test_predicted);
save('data\results\stats_alexnet_testing.mat','stats');
disp('Saved the Testing Results on Alexnet');