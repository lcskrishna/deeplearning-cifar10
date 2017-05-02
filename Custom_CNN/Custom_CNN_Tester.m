clear all;
close all;
clc;


[~,~,testing_data,testing_labels] = helperCIFAR10Data.load('./data');
disp('Done with forming the Testing Data:');

disp('Loading the Trained Custom CNN implemented:');
load('Pretrained_Networks\trained_cnn_validation_100_epoch.mat');
disp('Done loading the Custom CNN');

testing_output_predicted = classify(net,testing_data);
testing_accuracy = sum(testing_output_predicted==testing_labels)/numel(testing_labels);
disp('The validation Accuracy is :');
disp(testing_accuracy);

stats = confusionmatStats(testing_labels,testing_output_predicted);
save('./data/results/stats_testing.mat','stats');
disp('Done Saving the Performance Parameters for each class on Testing Dataset.');