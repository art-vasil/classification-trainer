import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pickle

dataset = pd.read_csv("dataset.csv",index_col=0)     # read the dataset create already
dataset = shuffle(dataset) # shuffle dataset

X = dataset.drop(["Value", "words"], axis=1)
y = dataset["Value"] 

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.33, random_state = 0)

##########Decision tree classifier####################################################3

from sklearn.tree import DecisionTreeClassifier
cpp = DecisionTreeClassifier()
cpp.fit(train_X, train_y)

decisionpred_y = cpp.predict(test_X)

cm = confusion_matrix(test_y, decisionpred_y)
print(cm)

pickle.dump(cpp, open("trained_decision.sav", 'wb'))  	# save trained decision tree model to directory

from sklearn.metrics import confusion_matrix, accuracy_score
#accuracy score
accuracy_score(test_y, decisionpred_y)

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(test_y, decisionpred_y)

print('precision: {}'.format(precision))#precision score
print('recall: {}'.format(recall))#recall score
print('fscore: {}'.format(fscore))#f-measure
print('support: {}'.format(support))

import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(test_y, decisionpred_y)
roc_auc = metrics.auc(fpr, tpr)

fpr, tpr, threshold = metrics.roc_curve(test_y, decisionpred_y)
roc_auc = metrics.auc(fpr, tpr)

# Plotting ROC curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

##########################Random forest##################################################3

from sklearn.ensemble import RandomForestClassifier
cll = RandomForestClassifier()
cll.fit(train_X, train_y)

randompred_y = cll.predict(test_X)		# perform prediction for evaluation

cmrand = confusion_matrix(test_y, randompred_y)
print(cmrand)

pickle.dump(cll, open("trained_Random.sav", 'wb'))  	# save trained random forest model to directory


from sklearn.metrics import confusion_matrix, accuracy_score
#accuracy score
accuracy_score(test_y, randompred_y)

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(test_y, randompred_y)

print('precision: {}'.format(precision))#precision score
print('recall: {}'.format(recall))#recall score
print('fscore: {}'.format(fscore))#f-measure
print('support: {}'.format(support))

import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(test_y, randompred_y)
roc_auc = metrics.auc(fpr, tpr)

fpr, tpr, threshold = metrics.roc_curve(test_y, randompred_y)
roc_auc = metrics.auc(fpr, tpr)

# Plotting ROC curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


##########################  SVM  #######################

from sklearn.svm import SVC
clf = SVC()			# create SVM object from sklearn

clf.fit(train_X, train_y)			# start training on SVM
svmpred_y = clf.predict(test_X) 		# perform prediction on the SVM dataset

cmsvc = confusion_matrix(test_y, svmpred_y, labels=[True, False])  # create confusion matrix for evaluation
print(cmsvc)

pickle.dump(clf, open("trained_svm.sav", 'wb'))   # save trained SVM model to directory

from sklearn.metrics import confusion_matrix, accuracy_score
#accuracy score
accuracy_score(test_y, svmpred_y)

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(test_y, svmpred_y)

print('precision: {}'.format(precision))#precision score
print('recall: {}'.format(recall))#recall score
print('fscore: {}'.format(fscore))#f-measure
print('support: {}'.format(support))

import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(test_y, svmpred_y)
roc_auc = metrics.auc(fpr, tpr)

fpr, tpr, threshold = metrics.roc_curve(test_y, svmpred_y)
roc_auc = metrics.auc(fpr, tpr)

# Plotting ROC curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

########## Artificial neural netwoks ########################

import tensorflow as tf
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(train_X, train_y, batch_size = 40, epochs = 60)

y_pred = ann.predict(test_X)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(test_y, y_pred.round())
print(cm)
# accuracy_score(test_y, y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score
#accuracy score
accuracy_score(test_y, y_pred.round())

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(test_y, y_pred.round())

print('precision: {}'.format(precision))#precision score
print('recall: {}'.format(recall))#recall score
print('fscore: {}'.format(fscore))#f-measure
print('support: {}'.format(support))

import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred.round())
roc_auc = metrics.auc(fpr, tpr)

fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred.round())
roc_auc = metrics.auc(fpr, tpr)

# Plotting ROC curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

