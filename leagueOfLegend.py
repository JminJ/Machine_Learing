import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

def readCsv(address):
    rdC = pd.read_csv('address')
    return rdC

dataset = pd.read_csv('D:\machineLearing_study\high_diamond_ranked_10min.csv')
corr = dataset.corr()
#print(corr['blueWins'].sort_values(ascending=False))

""" attribute = ['blueGoldDiff', 'blueExperienceDiff','blueGoldPerMin','blueTotalGold']
sns.pairplot(dataset[attribute], diag_kind='hist')
plt.show() """

scaler = StandardScaler()
X = scaler.fit_transform(dataset.drop(['blueWins'], axis = 1))
y = dataset.iloc[:,1]
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

pca = PCA(n_components = 20)
pca_xtrain = pca.fit_transform(x_train)
pca_xtest = pca.fit_transform(x_test)

linear_clf = LinearSVC()
param_grid = {
    'c' : []
}

ran_clf = RandomForestClassifier()
param_grid = {
    'n_estimators' : [2,5,10,15,20,25,30],
    'max_leaf_nodes' : [2,4,6,8,10],
    'max_depth' : [2,4],
    'max_features': [2,4,6,8,10]
}

grid_clf = GridSearchCV(estimator = ran_clf, param_grid = param_grid, cv = 10, verbose = 2, n_jobs=3)
grid_clf.fit(pca_xtrain, y_train)
print(grid_clf.best_params_)
print('------------------------------------------------------------------------')
print(grid_clf.best_score_)


