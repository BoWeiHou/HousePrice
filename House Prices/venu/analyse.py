# 不谷尉
import pandas as pd
import warnings

from venu.Simulate import clf

train_data = pd.read_csv('F:\PycharmProjects\House Prices\Data\/train.csv')
test_data = pd.read_csv('F:\PycharmProjects\House Prices\Data\/test.csv')

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',
        'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
print(test_data[cols].isnull().sum())
print(test_data['GarageArea'].describe())
print(test_data['TotalBsmtSF'].describe())
test_data['GarageCars'].fillna(1.766118, inplace=True)
test_data['TotalBsmtSF'].fillna(1046.117970, inplace=True)
print(test_data[cols].isnull().sum())
x = test_data.values
y_te_pred = clf.predict(x)
print(y_te_pred)
print(y_te_pred.shape)
print(x.shape)




