clear all;
close all;
clc;

%%% Get the Data.
[train_data,train_labels,testing_data,testing_labels] = helperCIFAR10Data.load('./data');
[height,width,channels,num_samples] = size(train_data);

disp('Size of trainig Labels');
disp(size(train_labels));

disp('Size of the training Data is:');
disp(height);
disp(width);
disp(channels);
disp(num_samples);

disp('Size of the Testing Data is:');
disp(size(testing_data));

%%% Formation of the Training and the validation dataset for cross
%%% validation.
X_train = train_data(:,:,:,1:46000);
Y_train = train_labels(1:46000);

X_validation = train_data(:,:,:,46001:num_samples);
Y_validation = train_labels(46001:num_samples);


%%% Form the convolution layer.
layers = [ imageInputLayer([height width channels]);
    convolution2dLayer(5,32,'Stride',1,'Padding',2);
    reluLayer();
    maxPooling2dLayer(3,'Stride',2,'Padding',0);
    convolution2dLayer(5,32,'Stride',1,'Padding',2);
    reluLayer();
    maxPooling2dLayer(3,'Stride',2,'Padding',0);
    convolution2dLayer(5,64,'Stride',1,'Padding',2);
    reluLayer();
    maxPooling2dLayer(3,'Stride',2,'Padding',0);
    convolution2dLayer(5,64,'Stride',1,'Padding',2);
    reluLayer();
    maxPooling2dLayer(3,'Stride',2,'Padding',0);
    fullyConnectedLayer(64);
    reluLayer();
    fullyConnectedLayer(numel(unique(train_labels)));
    softmaxLayer();
    classificationLayer(); ];

%%% Form the Training Options.
options=trainingOptions('sgdm','Momentum',0.9,'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.1,'LearnRateDropPeriod', 8, ...
    'L2Regularization',0.004,'MaxEpochs',100,'MiniBatchSize',128,'Verbose',true);

%% Train the network and save the network.
net=trainNetwork(train_data,train_labels,layers,options);
save('./data/networks/trained_cnn_validation.mat','net');
disp('Done Training the Data on CIFAR 10');
