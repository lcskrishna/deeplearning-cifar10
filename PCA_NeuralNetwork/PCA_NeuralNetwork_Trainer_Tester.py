
# coding: utf-8

# In[1]:

import pickle


# In[2]:

def unpickle(file):
    with open(file,"rb") as f:
        dict = pickle.load(f,encoding='latin1')
    return dict


# In[3]:

data1= unpickle('../../Dataset/cifar-10-batches-py/data_batch_1')
X_train1 = data1['data']
Y_train1 = data1['labels']
print("Size of the Training Features in Fold 1:")
print(len(X_train1))
print("Size of the Testing Features in Fold 2:")
print(len(Y_train1))


# In[4]:

data2= unpickle('../../Dataset/cifar-10-batches-py/data_batch_2')
X_train2 = data2['data']
Y_train2 = data2['labels']
print("Size of the Training Features of Fold 2 : ")
print(len(X_train2))
print("Size of the Training Labels of Fold 2: ")
print(len(Y_train2))


# In[5]:

data3= unpickle('../../Dataset/cifar-10-batches-py/data_batch_3')
X_train3 = data3['data']
Y_train3 = data3['labels']
print("Size of the Training Features of Fold 3 : ")
print(len(X_train3))
print("Size of the Training Labels of Fold 3: ")
print(len(Y_train3))


# In[6]:

data4= unpickle('../../Dataset/cifar-10-batches-py/data_batch_4')
X_train4 = data4['data']
Y_train4 = data4['labels']
print("Size of the Training Features of Fold 4 : ")
print(len(X_train4))
print("Size of the Training Labels of Fold 4: ")
print(len(Y_train4))


# In[7]:

data5= unpickle('../../Dataset/cifar-10-batches-py/data_batch_5')
X_train5 = data5['data']
Y_train5 = data5['labels']
print("Size of the Training Features of Fold 5 : ")
print(len(X_train5))
print("Size of the Training Labels of Fold 5: ")
print(len(Y_train5))


# In[8]:

### Training Features.
import numpy as np
X_train = np.append(X_train1,X_train2,axis=0)
X_train = np.append(X_train,X_train3,axis=0)
X_train = np.append(X_train,X_train4,axis=0)
Y_train = np.append(Y_train1,Y_train2,axis=0)
Y_train = np.append(Y_train,Y_train3,axis=0)
Y_train = np.append(Y_train,Y_train4,axis=0)
print("Done with the formation of the Training Features.")
print("The dimensions of the Training Features:")
print(X_train.shape)

print("The dimensions of the Training Labels:")
print(Y_train.shape)


# In[9]:

X_validation = np.array(X_train5)
Y_validation = np.array(Y_train5)
print("Done with forming the Validation Dataset.")


# In[10]:

### Formation of the Testing Dataset.
testing_data = unpickle('../../Dataset/cifar-10-batches-py/test_batch')
X_test = testing_data['data']
Y_test = testing_data['labels']
print("Done with the formation of the Testing dataset.")
print("Size of the Testing Features: ")
print(len(X_test))
print("Size of the Testing Labels: ")
print(len(Y_test))


# In[12]:

import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')


# In[ ]:

X_pca = pca.fit(X) 


# In[ ]:

clf = MLPClassifier(hidden_layer_sizes=500, activation='relu', solver='sgd', alpha=0.0001,learning_rate='adaptive', learning_rate_init=0.001, max_iter=10000, verbose=False,  early_stopping=True);
clf = clf.fit(X_pca,Y_train)
print("Neural Network Training is done with Iterations 10000 and Hidden Layers 500")


# In[ ]:

### Validation Dataset.
output_predicted_validation = clf.predict(X_validation)
accuracy_testing = metrics.accuracy_score(output_predicted_validation,Y_validation)
print("Accuracy on the Validation Dataset is: ")
print(accuracy_testing* 100)


# In[ ]:

## Performance measures of the Code.
print(" The Classification report is : ")
print(metrics.classification_report(Y_validation,output_predicted_validation))

print("The confusion Matrix obtained is : ")
print(metrics.confusion_matrix(Y_validation, output_predicted_validation))


# In[ ]:

output_predicted = clf.predict(X_test)
accuracy_testing = metrics.accuracy_score(output_predicted,Y_test)
print("Accuracy on the Testing Dataset is: ")
print(accuracy_testing* 100)


# In[ ]:

### Performance Measures in the Code.
print("The Classification Report is: ")
print(metrics.classification_report(Y_test,output_predicted))

print("The Confusion Matrix to verify its performance is:")
print(metrics.confusion_matrix(Y_test,output_predicted))

