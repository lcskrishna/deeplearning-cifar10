
# coding: utf-8

# In[1]:

import pickle


# In[11]:

import pickle
with open("data_batch_1", "rb") as f:
    data1 = pickle.load(f, encoding='latin1')



# In[3]:

def unpickle(file):
    with open("data_batch_1", "rb") as f:
        dict = pickle.load(f, encoding='latin1')
    return dict


# In[5]:

data1= unpickle('data_batch_1')


# In[7]:

X_train1 = data1['data']
Y_train1 = data1['labels']
print("Size of the Training Features:")
print(len(X_train1))
print("Size of the Testing Features:")
print(len(Y_train1))


# In[8]:

data2= unpickle('data_batch_2')
X_train2 = data2['data']
Y_train2 = data2['labels']
print("Size of the Training Features of Fold 2 : ")
print(len(X_train2))
print("Size of the Training Labels of Fold 2: ")
print(len(Y_train2))



# In[16]:

data3= unpickle('data_batch_3')
X_train3 = data3['data']
Y_train3 = data3['labels']
print("Size of the Training Features of Fold 3 : ")
print(len(X_train3))
print("Size of the Training Labels of Fold 3: ")
print(len(Y_train3))


# In[17]:

data4= unpickle('data_batch_4')
X_train4 = data4['data']
Y_train4 = data4['labels']
print("Size of the Training Features of Fold 4 : ")
print(len(X_train4))
print("Size of the Training Labels of Fold 4: ")
print(len(Y_train4))


# In[18]:

data5= unpickle('data_batch_5')
X_train5 = data5['data']
Y_train5 = data5['labels']
print("Size of the Training Features of Fold 5 : ")
print(len(X_train5))
print("Size of the Training Labels of Fold 5: ")
print(len(Y_train5))


# In[20]:

### Training Features.
import numpy as np
X_train = np.append(X_train1,X_train2,axis=0)
X_train = np.append(X_train,X_train3,axis=0)
X_train = np.append(X_train,X_train4,axis=0)
Y_train = np.append(Y_train1,Y_train2,axis=0)
Y_train = np.append(Y_train,Y_train3,axis=0)
Y_train = np.append(Y_train,Y_train4,axis=0)
print("The dimensions of the Training Features:")
print(X_train.shape)

print("The dimensions of the Training Labels:")
print(Y_train.shape)


# In[21]:

#Validation Features:
X_validation = np.array(X_train5)
Y_validation = np.array(Y_train5)
print("The dimensions of the Validation Features: ")
print(X_validation.shape)
print("The dimensions of the Validation Labels: ")
print(Y_validation.shape)


# In[ ]:

########### Implementation of the KNN Model.
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

clf = KNeighborsClassifier()
clf = clf.fit(X_train,Y_train)
print("Classification is Done.")


# In[ ]:

output_Predicted = clf.predict(X_train);
accuracy_training = metrics.accuracy_score(output_Predicted,Y_train)
print("Accuracy on the Training Data set:")
print(accuracy_training* 100)

output_predicted_validation = clf.predict(X_validation)
accuracy_2ndFold = metrics.accuracy_score(output_predicted_validation,Y_validation)
print("Accuracy on the Validation Data set is:")
print(accuracy_2ndFold * 100)

