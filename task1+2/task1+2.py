
# coding: utf-8

# Task 1: Analyze distances between images
# 
# The purpose of this task is to develop some intuitions about clouds of points in highly
# dimensional spaces. In particular, you are supposed to develop a very simple algorithm for
# classifying hand-written digits.
# 
# Let us start with developing a simple distance-based classifier. For each digit d, d =
# 0, 1, ... 9, let us consider a cloud of points in 256 dimensional space, Cd, which consists of
# all training images (vectors) that represent d. Then, for each cloud Cd we can calculate its
# center, cd, which is just a 256-dimensional vector of means over all coordinates of vectors that
# belong to Cd. Once we have these centers, we can easily classify new images: to classify an
# image, calculate the distance from the vector that represents this image to each of the 10
# centers; select as a label the closest one. 
# 
# But first let us take a closer look at out data.
# For each cloud Cd, d = 0, 1, ... 9, calculate its center, ci, and the radius, ri. The radius of
# Cd is defined as the biggest distance between the center of Cd and points from Cd. Addition-
# ally, find the number of points that belong to Ci, ni. Clearly, at this stage you are supposed
# to work with the training set only.
# 
# Next, calculate the distances between the centers of the 10 clouds, distij = dist(ci; cj ), for
# i; j = 0, 1, ... 9. Given all these distances, try to say something about the expected accuracy
# of your classifier. What pairs of digits seem to be most difficult to separate?

# In[2]:


import os
import csv
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise


# In[3]:


#set directory
#import data
os.chdir('../data')
trainin = pd.read_csv('train_in.csv', header = None)
trainout = pd.read_csv('train_out.csv', header = None)
testin = pd.read_csv('test_in.csv', header = None)
testout = pd.read_csv('test_out.csv', header = None)


# In[13]:


#calculate the centers
#matrix representing the center points, each row represent each digit(0-9)
Cs=np.zeros((10,256))

#sum up the values
for i in range(0, trainin.shape[0]):
    num = trainout.iloc[i,0]
    Cs[num,] += trainin.iloc[i,]
    
#number of each digits
Ns = trainout.groupby(0).size()

#average
for i in range(0,10):
    Cs[i,] = Cs[i,]/Ns[i]

#calculate radius
Rs = [0]*10
for i in range(0, 10):
    idx = np.where(trainout == i)[0]
    for id in idx:
        d = np.linalg.norm(Cs[i,] - trainin.iloc[id,]) #Euclidean norm distance
        if d>Rs[i]:
            Rs[i] = d

#calculate distances between the centers
Ds = np.zeros((10,10))
for i in range(0,10):
    for j in range(0, 10):
        Ds[i,j] = np.linalg.norm(Cs[i,] - Cs[j,])


# In[5]:


# take a look at the distances
pd.DataFrame(Ds)

#dist(3,5) = 6.118 
#dist(4,9) = 6.010
#dist(7,9) - 5.426
#the most difficult pairs to tell apart


# In[15]:


Rs = [ '%.2f' % elem for elem in Rs]
Rs
#Radius shows how far the outliers are from the center, for each category
#for almost all categories except 2, all the centers of other categories are within the radius


# Task 2: Implement and evaluate the simplest classifier
# 
# Implement the simplest distance-based classifier that was described above. Apply your
# classifier to all points from the training set and calculate the percentage of correctly classified
# digits. Do the same with the test set, using the centers that were calculated from the training
# set. In both cases, generate a confusion matrix which should provide a deeper insight into
# classes that are difficult to separate. A confusion matrix is here a 10-by-10 matrix (cij ),
# where cij contains the percentage (or count) of digits i that were classified as j. Which digits
# were most difficult to classify correctly? For calculating and visualising confusion matrices
# you may use the sklearn package. Describe your findings. Compare performance of your
# classifier on the train and test sets. How do the results compare to the observations you've
# made in Step 1? How would you explain it?
# 
# So far, we have implicitly assumed that you have measured the distance between
# two vectors with help of the Euclidean distance measure. However, this is not the only
# choice. Rerun your code using alternative distance measures that are implemented in
# sklearn.metrics.pairwise.pairwise distances. Which distance measure provides best
# results (on the test set)?

# In[39]:


#define functions
def center_dist_classifier(datamat, class_nr, centers, dist_measure):
    class_dist = np.zeros((datamat.shape[0], class_nr))
    for i in range(0,datamat.shape[0]):
        for j in range(0,class_nr):
            class_dist[i,j] = float(pairwise.pairwise_distances([centers[j,]], [datamat.iloc[i,]], metric=dist_measure))
    est_class = pd.DataFrame(class_dist).idxmin(axis=1)
    return(est_class.values)

#normalization and visualization of the confusion matrix
#source:http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[40]:


#classify training set, Euclidean distance measure
ecd_train = center_dist_classifier(trainin, 10, Cs, 'euclidean') 
trainout_array = [item for sublist in trainout.values for item in sublist]
ecd_train_cm = confusion_matrix(trainout_array, ecd_train)

plt.figure()
plot_confusion_matrix(ecd_train_cm, classes=['0','1','2','3','4','5','6','7','8','9'], 
                      normalize=True, title='Euclidean, Train set')
plt.show()


# In[26]:


#overall accuracy
np.mean(ecd_train == trainout_array)


# In[42]:


#classify test set, Euclidean distance measure
ecd_test = center_dist_classifier(testin, 10, Cs, 'euclidean') 
testout_array = [item for sublist in testout.values for item in sublist]
ecd_test_cm = confusion_matrix(testout_array, ecd_test)

plt.figure()
plot_confusion_matrix(ecd_test_cm, classes=['0','1','2','3','4','5','6','7','8','9'], 
                      normalize=True, title='Euclidean, Test Set')
plt.show()


# In[43]:


#overall accuracy
np.mean(ecd_test == testout_array)


# In[45]:


Ns


# In[46]:


#classify test set, cosine measure
cos_test = center_dist_classifier(testin, 10, Cs, 'cosine') 
cos_test_cm = confusion_matrix(testout_array, cos_test)

plt.figure()
plot_confusion_matrix(cos_test_cm, classes=['0','1','2','3','4','5','6','7','8','9'], 
                      normalize=True, title='Cosine, Test Set')
plt.show()


# In[47]:


#overall accuracy
np.mean(cos_test == testout_array)


# In[ ]:


#overall accuracy
np.mean(ecd_train == trainout_array)


# In[48]:


#classify test set, Manhattan measure
mh_test = center_dist_classifier(testin, 10, Cs, 'manhattan') 
mh_test_cm = confusion_matrix(testout_array, mh_test)

plt.figure()
plot_confusion_matrix(mh_test_cm, classes=['0','1','2','3','4','5','6','7','8','9'], 
                      normalize=True, title='Manhattan, Test Set')
plt.show()


# In[49]:


#overall accuracy
np.mean(mh_test == testout_array)


# In[109]:


