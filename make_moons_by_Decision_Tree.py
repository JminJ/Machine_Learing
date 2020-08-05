from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

dataset = list(make_moons(n_samples=1000, noise=0.4)) # make_moons 데이터 적재
dataset_x = dataset[0] 
dataset_y = str(dataset[1]) # int형에서 str형으로 변환

dataset_Y = [] # dataset_y값을 가공한 결과 값
for i in dataset_y[1:-1]:
  if ord(i) >= 48 and ord(i) <= 57: # <ASCII코드> 숫자인지 검사
    dataset_Y.append(int(i)) # 숫자가 맞으면 dataset_Y에 값 추가

X_train, X_test, Y_train, Y_test = train_test_split(dataset_x,dataset_Y, random_state = 48)

param_grid = {
    'max_depth' : [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'max_leaf_nodes' : [3,4,5,6,7,8,9,10,11,12,13,14,15]
}

cv = KFold(random_state= 48, n_splits= 10) # 교차 검증

estimator = DecisionTreeClassifier() # 사용될 알고리즘

grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid, cv = cv, verbose= 2) 
# 최적의 DecicionTreeClassifier 하이퍼 파라미터를 찾아 준다

grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)#max_depth = 4, max_leaf_nodes = 15(random_state = 18, n_splits = 6)
                                            
print('--------------------------------------------------------------------')
print(grid_search.best_score_)


