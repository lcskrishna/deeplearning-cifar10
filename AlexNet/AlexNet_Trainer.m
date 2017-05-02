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

disp('Done loading the Training Dataset');


%% Define a Alexnet architecture for CIFAR 10 data.
conv1 = convolution2dLayer(5,32,'Padding',2,...
                     'BiasLearnRateFactor',2);
% conv1.Weights = gpuArray(single(randn([5 5 3 32])*0.0001));
fc1 = fullyConnectedLayer(64,'BiasLearnRateFactor',2);
% fc1.Weights = gpuArray(single(randn([64 576])*0.1));
fc2 = fullyConnectedLayer(10,'BiasLearnRateFactor',2);
% fc2.Weights = gpuArray(single(randn([10 64])*0.1));

layers = [ ...
    imageInputLayer([32 32 3]);
    conv1;
    maxPooling2dLayer(3,'Stride',2);
    reluLayer();
    convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    fc1;
    reluLayer();
    fc2;
    softmaxLayer()
    classificationLayer()];
 
 %%% Form the Training Options.
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 100, ...
    'Verbose', true);

disp('Started Training Data');

%% Train the network and save the network.
net=trainNetwork(train_data,train_labels,layers,options);
save('./data/networks/trained_alexnet_validation.mat','net');
disp('Done Training the Data on CIFAR 10');
