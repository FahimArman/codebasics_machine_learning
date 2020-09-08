from sklearn import datasets,svm
import pandas as pd
iris = datasets.load_iris()

from sklearn.model_selection import GridSearchCV

clf= GridSearchCV(svm.SVC(gamma='auto'),{
	'C':[1,10,20],
	'kernel':['rbf','linear']
}, cv = 5, return_train_score = False)

clf.fit(iris.data,iris.target)

#print(clf.cv_results_)
clf.cv_results_
print(clf.best_score_)



df = pd.DataFrame(clf.cv_results_)
print(df[['param_C','param_kernel','mean_test_score']])
#print(dir(clf))
print(clf.best_score_)
print(clf.best_params_)
