Section 1: Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Section 2: Reading and Preprocessing Data
#Reading the csv files
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
stores = pd.read_csv('stores.csv')
features = pd.read_csv('features.csv')

#Merging the stores and features dataframe with the train and test dataframes
train_data = train_data.merge(stores, on='Store').merge(features, on=['Store','Date'])
test_data = test_data.merge(stores, on='Store').merge(features, on=['Store','Date'])

#Converting Date column to datetime format
train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])

#Section 3: Exploratory Data Analysis
#Plotting the distribution of Weekly Sales in the train data
sns.histplot(train_data['Weekly_Sales'], kde=False, bins=50)
plt.title('Distribution of Weekly Sales')
plt.xlabel('Weekly Sales')
plt.ylabel('Frequency')
plt.show()

#Plotting the correlation matrix
corr = train_data.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.title('Correlation Matrix')
plt.show()

#Section 4: Feature Engineering
#Creating a new feature 'Week_Number' from Date
train_data['Week_Number'] = train_data['Date'].dt.week
test_data['Week_Number'] = test_data['Date'].dt.week

#Creating a new feature 'Month' from Date
train_data['Month'] = train_data['Date'].dt.month
test_data['Month'] = test_data['Date'].dt.month

#Creating a new feature 'Year' from Date
train_data['Year'] = train_data['Date'].dt.year
test_data['Year'] = test_data['Date'].dt.year

#Creating a new feature 'IsHoliday' from the 'Holiday_Flag' column in features dataframe
train_data['IsHoliday'] = train_data['IsHoliday_x'] | train_data['IsHoliday_y']
test_data['IsHoliday'] = test_data['IsHoliday_x'] | test_data['IsHoliday_y']

#Section 5: Building Model
#Splitting the train data into features and target variable
X_train = train_data.drop(['Weekly_Sales', 'Date'], axis=1)
y_train = train_data['Weekly_Sales']

#Splitting the test data into features and target variable
X_test = test_data.drop(['Date'], axis=1)

#Scaling the features using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Training the model using XGBRegressor
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train_scaled, y_train)

#Section 6: Prediction and Evaluation
Predicting the target variable using the test data
y_pred = model.predict(X_test_scaled)

#Creating the submission file
submission = pd.DataFrame({
'Store': test_data['Store'],
'Dept': test_data['Dept'],
'Date': test_data['Date'],
'Weekly_Sales': y_pred
})

submission.to_csv('submission.csv', index=False)