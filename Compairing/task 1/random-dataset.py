#import the necessary models
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pickle

dataset = pd.read_csv("rand_data.csv",index_col=0)     # read the dataset create already
dataset = shuffle(dataset)			 # shuffle dataset

# split dataset into train and test			 
split_idx = int(0.9*len(dataset))	#multiply dataset length by 0.91 
train_data = dataset[:split_idx].copy()
test_data = dataset[split_idx:].copy()
dataset = None

print(train_data['__label__'].value_counts())
print(test_data['__label__'].value_counts())


train_X = train_data.drop('__label__', axis=1)	# remove the labels column from the dataset as it's not needed for training
train_y = train_data['__label__']
train_data.shape, train_X.shape, train_y.shape

############# DecisionTreeClassifier

# import model

#hyperparameters
from sklearn import tree
Hyper_1 = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=3)

from sklearn.tree import DecisionTreeClassifier
cpp = DecisionTreeClassifier()
cpp.fit(train_X, train_y) # fit the DecisionTreeClassifier in train_X, train_y

test_X = test_data.drop("__label__", axis=1)   # remove the labels column from the dataset as it's not needed for training
test_y = test_data['__label__']
decisionpred_y = cpp.predict(test_X)		# perform prediction for evaluation

# perform confusion matrix to find out false positive, false negative, true positive, true negative values
cm = confusion_matrix(test_y, decisionpred_y, labels=[True, False])
print(cm)

pickle.dump(cpp, open("trained_decision.sav", 'wb'))  	# save trained decision tree model to directory


from sklearn.metrics import confusion_matrix, accuracy_score
#accuracy score
accuracy_score(test_y, decisionpred_y)

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(test_y, decisionpred_y)

print('precision: {}'.format(precision)) #precision score
print('recall: {}'.format(recall)) #recall score
print('fscore: {}'.format(fscore)) #f-measure
print('support: {}'.format(support))

#ROC curve
import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(test_y, decisionpred_y) #roc curve on fpr and tpr
roc_auc = metrics.auc(fpr, tpr)

fpr, tpr, threshold = metrics.roc_curve(test_y, decisionpred_y)
roc_auc = metrics.auc(fpr, tpr)

# Plotting ROC curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)  #plotting fpr and tpr
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

################## RandomForestClassifier

#Hyperparameters
from sklearn.model_selection import RandomizedSearchCV
random_grid = {'bootstrap': [True, False],
'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
'max_features': ['auto', 'sqrt'],
'min_samples_leaf': [1, 2, 4],
'min_samples_split': [2, 5, 10],
'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

from sklearn.ensemble import RandomForestClassifier
cll = RandomForestClassifier()
cll.fit(train_X, train_y)

test_X = test_data.drop("__label__", axis=1)   # remove the labels column from the dataset as it's not needed for training
test_y = test_data['__label__']
randompred_y = cll.predict(test_X)		# perform prediction for evaluation

# perform confusion matrix to find out false postive, false negetive, true positive, true negetive values
cm = confusion_matrix(test_y, randompred_y, labels=[True, False])
print(cm)

pickle.dump(cll, open("trained_Random.sav", 'wb'))  	# save trained Naive Bayes model to directory

from sklearn.metrics import confusion_matrix, accuracy_score
#accuracy score

accuracy_score(test_y, randompred_y)

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(test_y, randompred_y)

print('precision: {}'.format(precision))#precision score
print('recall: {}'.format(recall))# recall score
print('fscore: {}'.format(fscore))# f-measure
print('support: {}'.format(support))

#ROC curve
import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(test_y, randompred_y) #roc curve on fpr and tpr
roc_auc = metrics.auc(fpr, tpr)

fpr, tpr, threshold = metrics.roc_curve(test_y, randompred_y)
roc_auc = metrics.auc(fpr, tpr)

# Plotting ROC curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)  #plotting fpr and tpr
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

####################### SVM ##########################

test_X = test_data.drop("__label__", axis=1)
test_y = test_data['__label__']

#Hyperparameters

from sklearn.svm import SVC
clf = SVC(C=1, kernel='rbf', gamma=0.01)			# create SVM object from sklearn

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
print('recall: {}'.format(recall))# recall score
print('fscore: {}'.format(fscore))# f-measure
print('support: {}'.format(support))

#roc
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




###################### KNeighborsClassifier

#Hyperparameters
n, p = 20, 2

from sklearn.neighbors import KNeighborsClassifier
crr = KNeighborsClassifier()
crr.fit(train_X, train_y)

test_X = test_data.drop("__label__", axis=1)  # remove the labels column from the dataset as it's not needed for training
test_y = test_data['__label__']
knearestpred_y = crr.predict(test_X)		# perform prediction for evaluation

# perform confusion matrix to find out false postive, false negetive, true positive, true negetive values
cm = confusion_matrix(test_y, knearestpred_y, labels=[True, False])
print(cm)

pickle.dump(crr, open("trained_knearest.sav", 'wb'))  	# save trained Naive Bayes model to directory

from sklearn.metrics import confusion_matrix, accuracy_score
#accuracy score

accuracy_score(test_y, knearestpred_y)

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(test_y, knearestpred_y)

print('precision: {}'.format(precision))#precision score
print('recall: {}'.format(recall))#recall score
print('fscore: {}'.format(fscore))#f-measure
print('support: {}'.format(support))

#ROC curve
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(test_y, knearestpred_y) #roc curve on fpr and tpr
roc_auc = metrics.auc(fpr, tpr)

fpr, tpr, threshold = metrics.roc_curve(test_y, knearestpred_y)
roc_auc = metrics.auc(fpr, tpr)

# Plotting ROC curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc) #plotting fpr and tpr
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


####################### Naive Bayes ##################

from sklearn.model_selection import RepeatedStratifiedKFold
cv_method = RepeatedStratifiedKFold(n_splits=20, n_repeats=15, random_state = 2000)

from sklearn.naive_bayes import GaussianNB
cjj = GaussianNB()					# create Naive Bayes object from skelarn
cjj.fit(train_X, train_y)			# start training on Naive Bayes

test_X = test_data.drop("__label__", axis=1)
test_y = test_data['__label__']
naivepred_y = cjj.predict(test_X)		# perform prediction for evaluation


cm = confusion_matrix(test_y, naivepred_y, labels=[True, False])
print(cm)

pickle.dump(cll, open("trained_nb.sav", 'wb'))  	# save trained Naive Bayes model to directory

from sklearn.metrics import confusion_matrix, accuracy_score

accuracy_score(test_y, naivepred_y)

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(test_y, naivepred_y)

print('precision: {}'.format(precision))#precision score
print('recall: {}'.format(recall))# recall score
print('fscore: {}'.format(fscore))# f-measure
print('support: {}'.format(support))

#ROC curve
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(test_y, naivepred_y)
roc_auc = metrics.auc(fpr, tpr)

fpr, tpr, threshold = metrics.roc_curve(test_y, naivepred_y)
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
