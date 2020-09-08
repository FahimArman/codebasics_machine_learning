from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_digits

digits = load_digits()


from sklearn.model_selection import cross_val_score

print(cross_val_score(LogisticRegression(),digits.data,digits.target,cv=3))
print(cross_val_score(SVC(),digits.data,digits.target,cv=3))
print(cross_val_score(RandomForestClassifier(n_estimators=40),digits.data,digits.target,cv=3))








