# 不谷尉
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import  explained_variance_score, r2_score


# clfs = {
#         'svm':svm.SVR(),
#         'RandomForestRegressor':RandomForestRegressor(n_estimators=400),
#         'BayesianRidge':linear_model.BayesianRidge()
#        }
# for clf in clfs:
#     try:
#         clfs[clf].fit(X_train, y_train)
#         y_pred = clfs[clf].predict(X_test)
#         print(clf + " cost:" + str(np.sum(y_pred-y_test)/len(y_pred)) )
#     except Exception as e:
#         print(clf + " Error:")
#         print(str(e))


rcParams['font.sans-serif'] = 'SimHei'

train_data = pd.read_csv('F:\PycharmProjects\House Prices\Data\/train.csv')
test_data = pd.read_csv('F:\PycharmProjects\House Prices\Data\/test.csv')

cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
X = train_data[cols].values
print(X)
y = train_data['SalePrice'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestRegressor(n_estimators=400)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print('y-pred', y_pred)
fig = plt.figure(figsize=(10, 6))
plt.plot(range(y_test.shape[0]), y_test, color='blue', linewidth=1.5, linestyle='-')
plt.plot(range(y_test.shape[0]), y_pred, color='red', linewidth=1.5, linestyle='-')
plt.legend(['真实值', '预测值'])
plt.show()
print(r2_score(y_test, y_pred))
print(explained_variance_score(y_test, y_pred))


