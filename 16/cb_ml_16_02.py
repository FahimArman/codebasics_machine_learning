from sklearn import datasets,svm
import pandas as pd
iris = datasets.load_iris()

from sklearn.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(svm.SVC(gamma='auto'),{
	'C':[1,10,20],
	'kernel': ['rbf','linear']
	},
	cv = 5,
	return_train_score = False,
	n_iter =  2
)
rs.fit(iris.data, iris.target)

df = pd.DataFrame(rs.cv_results_)
print(df[['param_C','param_kernel','mean_test_score']])
#print(dir(clf))
print(rs.best_score_)
print(rs.best_params_)
