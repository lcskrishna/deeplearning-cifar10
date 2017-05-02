Title : Object Detection and Image Classification using CNN on CIFAR 10.
---------------------------------------------

The below procedure explains how to run the given code :
#########################################################################

Pre requisites:
1. System with Anaconda software installed
Link: https://www.continuum.io/downloads

To open your workspace type the following and navigate to the folder where you have kept the code.
jupyter notebook

2. System with Matlab 2017a version installed.

#########################################################################

Code Structure:

	AlexNet/
		-- data/results/
		      -- stats_alexnet_testing.mat
			  -- stats_alexnet_validation.mat
		-- Logs/out_train_alexnet_cifar10_875643.cph-m1.uncc.edu
		-- AlexNet_Tester.m
		-- AlexNet_Trainer.m
		-- AlexNet_Validator.m
		-- confusionmatStats.m
	
	Custom_CNN/
		-- data/results/
		      -- stats_testing.mat
			  -- stats_validation.mat
		-- Logs/out_train_Validation_cifar10_873177.cph-m1.uncc.edu
		-- CustomCNN_Tester.m
		-- CustomCNN_Trainer.m
		-- CustomCNN_Validator.m
		-- confusionmatStats.m
	
	Dataset/
		-- Matlab/cifar-10-batches-mat/
		-- Python/cifar-10-batches-py/
		
	KNN/
		--Logs/out_knn_test_knn_cifar_879323.cph-m1.uncc.edu
		--KNN_Analysis.py
		--KNN_Tester.py
		
	NeuralNetworks/
		-- Logs/out_nn_train_sgd_nn_cifar_882619.cph-m1.uncc.edu
		--NeuralNetwork_Trainer_Tester.py
		
	PCA_NeuralNetworks/
		-- Logs/out_nn_train_pca_nn_cifar_882632.cph-m1.uncc.edu
		--PCA_NeuralNetwork_Trainer_Tester.py

	TrainedModels/
		-- AlexNet/trained_alexnet_validation.mat
		-- Custom_CNN/trained_cnn_validation.mat
	
	README.txt 
###########################################################################

To run the program:
1) For KNN, Neural Networks and PCA_Neural Networks (Traditional Methods)
   use anaconda software.
2) Open Jupyter notebook in command prompt and run the program.

3) For AlexNet, Custom_CNN, Use Matlab2017a. 
   To run the program consider the pre trained models in TrainedModels folder and run the AlexNet_Tester.m or CustomCNN_Tester.m
   This gives the testing results and saves the results in data/results folder.
   
   To verify the results just load the mat file in the above data/results folder and get the respective Performance measure : FScore, Confusion Matrix, Precision etc.
   There are a total of 10 performance measures calculated for each of the alexnet and custom cnn.
  

###########################################################################
