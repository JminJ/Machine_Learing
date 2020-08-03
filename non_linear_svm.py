from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt

def plot_dataset(X,y, axes):
    plt.plot(X[:, 0][y == 0], X[:,1][y == 0], 'bs')
    plt.plot(X[:, 0][y == 1], X[:,1][y == 1], 'g^')
    plt.axis(axes)
    plt.grid(True, which = 'both')
    plt.xlabel(r'X1', fontsize = 20)
    plt.ylabel(r'X2', fontsize = 20, rotation = 0)


def plot_predictions(clf, axes):
    x0a = np.linspace(axes[0],axes[1], 100)
    x1a = np.linspace(axes[0],axes[3], 100)
    x0, x1 = np.meshgrid(x0a, x1a)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contour(x0, x1, y_pred, cmap = plt.cm.brg, alpha = 0.2)
    plt.contour(x0, x1, y_decision, cmap = plt.cm.brg, alpha = 0.1)

X, y = make_moons(n_samples=100, noise=0.15)
PolynomialFeatures_svm_clf = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler',StandardScaler()),
    ('svm_clf',LinearSVC(C = 10, loss = 'hinge'))
])
PolynomialFeatures_svm_clf.fit(X,y)

plot_predictions(PolynomialFeatures_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X,y,[-1.5, 2.5, -1, 1.5])
plt.show()