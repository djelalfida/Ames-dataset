import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px



# Import csv file.
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Replace empty string or NA with np.nan
# df = df.replace(['NA', ''], np.nan)

def clean_data(df):
  # Dropping the columns with more than 1179 null values
  df.drop(['PoolQC', 'Fence', 'Alley', 'MiscFeature', 'Electrical'], axis=1, inplace=True)
  # Drop Id because we don't need it
  # df.drop('Id', axis=1, inplace=True)


  df.replace('NA', np.nan, inplace=True)


  # Replace typos in data
  words_to_replace = {
      "Twnhs": "TwnhsE",
      "NAmes": "Names",
      "Wd Shng": "WdShing",
      "CmentBd": "CemntBd",
      "Brk Cmn": "BrkComm"
  }

  # Replace typos
  for key, value in df.items():
    df.replace(key, value, inplace=True)


  # Array with numerical missing values
  numerical_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

  # Instantiate median imputer
  imputer = SimpleImputer(missing_values=np.nan, strategy='median')

  # Replace np.nan values with the median of that column
  for feature_name in numerical_features:
    imputer = imputer.fit(df[[feature_name]])
    df[feature_name] = imputer.transform(df[[feature_name]]).ravel()


  # Categorical features with missing data
  categorical_features = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MSZoning']

  # Instantiate most_frequent imputer
  imputer_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

  for feature_name in categorical_features:
    imputer_freq = imputer_freq.fit(df[[feature_name]])
    df[feature_name] = imputer_freq.transform(df[[feature_name]]).ravel()


  categorical_features = ["BsmtFinType1", "BsmtFinType2", "BsmtExposure", "BsmtCond", 
                  "BsmtQual", "Foundation", "ExterCond", "ExterQual", 
                  "MasVnrType", "Exterior2nd", "Exterior1st", "RoofMatl", 
                  "RoofStyle", "HouseStyle", "BldgType", "Condition2", 
                  "Condition1", "Neighborhood", "HeatingQC", "Heating",
                  "SaleCondition", "SaleType", "PavedDrive", "GarageCond",
                  "GarageQual", "GarageFinish", "GarageType", "FireplaceQu",
                  "Functional", "KitchenQual", "CentralAir", "LandSlope",
                  "LotConfig", "Utilities", "LandContour", "LotShape",
                  "Street", "MSZoning", 
                  ]


  # Get dummies for categorical data
  df = pd.get_dummies(df, categorical_features)

  return df

# Clean data
data = clean_data(data)

# Test data
test_data = clean_data(test)

# make target y the SalePrice
y = data.SalePrice

# Drop SalePrice and assign the rest to X
X = data.drop('SalePrice', axis=1)

# Instantiate reg
reg = LinearRegression()

# Split into validation and training data.
X_train, X_test, y_train, y_test = train_test_split(X[test_data.columns], y, test_size = 0.3, random_state=42)

# Fit model on the training data.
reg.fit(X_train, y_train)

# Print score
print(reg.score(X_train, y_train)) # 0.9140314530388186

# Replace nan with 0 in test data
test_data.replace(np.nan, 0, inplace=True)

# Predict SalePrice
pred = reg.predict(test_data)

# Add feature SalePrice with the predicted SalePrice
test_data['SalePrice'] = pred

# New df with Id & SalePrice (Kaggle)
submission = test_data[['Id', 'SalePrice']]
submission.to_csv('submission.csv', index=False)
