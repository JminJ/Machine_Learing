import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('D:\machineLearing_study\high_diamond_ranked_10min.csv')
corr = dataset.corr()

scaler = StandardScaler()
X = scaler.fit_transform(dataset.drop(['gameId','blueWins'], axis = 1))
y = dataset.iloc[:,1]
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

pca = PCA(n_components = 20)
pca_xtrain = pca.fit_transform(x_train)
pca_xtest = pca.fit_transform(x_test)

param_grid = {
    #'n_estimators' : [10,100,500,1000],
    'max_depth' : [5, 10, 15, 20],
    'max_features' : [6, 12, 18, 24],
    'max_leaf_nodes' : [15, 30, 45],
    'min_samples_leaf' : [5, 8, 10]
}
decision_clf = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=decision_clf, param_grid=param_grid, cv = 15, n_jobs = 4)


param_grid2 = {
    'n_estimators' : [2,5,10,15,20,25],
    'max_leaf_nodes' : [2,4,6,8,10],
    'min_samples_leaf' : [2,3],
    'max_depth' : [2,4],
    'max_features': [2,4,6,8,10]
}
ran_clf = RandomForestClassifier()
grid_search2 = GridSearchCV(estimator = ran_clf, param_grid = param_grid2, cv = 10, verbose = 2, n_jobs=4)


param_grid3 = {
    'C' : [0.001, 0.1, 1, 10, 100], #0.001 = 73%
    'kernel' : ['linear']
}
svc_clf = SVC()
grid_search3 = GridSearchCV(estimator = svc_clf, param_grid = param_grid3, cv = 10, n_jobs = 4)

votingC = VotingClassifier(estimators  = [('linear',grid_search),('randomF',grid_search2)], voting='soft', n_jobs=4)
votingC = votingC.fit(x_train, y_train)
voReC = votingC.predict(x_test)
print(accuracy_score(y_test, voReC))
