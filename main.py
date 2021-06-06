# importing necessary modules
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, mean_squared_error

# Reading the data from white wine dataset
data = pd.read_csv('winequality-white.csv', delimiter=';')

# Reading the data from red wine dataset
data2 = pd.read_csv('winequality-red.csv', delimiter=';')

# Inserting 'type' column with white and red wine
data.insert(0, 'type', 'white')
data2.insert(0, 'type', 'red')

# Appending both data and data2 column-wise
df = data.append(data2, ignore_index = True)

# Calculating the Correlation Matrix
corr_mat = df.corr()
plt.figure(figsize=[20,10])
sns.heatmap(corr_mat,annot=True)

# Identifying the column which is highly co-related to other columns
for i in range(len(corr_mat.columns)):
    for j in range(i):
        if abs(corr_mat.iloc[i,j]) > 0.7:
            col_name = corr_mat.columns[i]
            print(col_name)

# Dropping the 'total sulfur dioxide' column
df.drop('total sulfur dioxide',axis=1, inplace = True)

# For handling of categorical columns
df = pd.get_dummies(df,drop_first=True)

# Considering the wine of best quality only if the quality of the wine is greater than 7
df['best quality'] = [ 1 if x>=7 else 0 for x in df.quality] 

# Dropping the 'quality' column
df.drop(['quality'], axis=1, inplace=True)

# Segregating the target variable i.e. 'best quality'
y = df['best quality']

# Considering the dependent variables from the dataset
x = df.drop(['best quality'], axis=1)

# Splitting the data into train and test sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# Randomized Search CV Parameters
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

# Creating the grid
grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Instantiating Random Forest Classifier
ran_forest = RandomForestClassifier()

rf = RandomizedSearchCV(estimator = ran_forest, param_distributions = grid, scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, n_jobs = 1)

rf.fit(x_train,y_train)

# Predicting the test dataset
predictions = rf.predict(x_test)

# Calculating the score of the model on test set
model_score = rf.score(x_test,y_test)

# Calculating Mean Squared Error
MSE = mean_squared_error(y_test,predictions)

# Calculating Root Mean Squared Error
RMSE = np.sqrt(MSE)

print('Mean Squared Error is : ', MSE)
print('Root Mean Squared Error is : ', RMSE)
print('Score of the model is : ', model_score)
print(classification_report(y_test, predictions))

# Saving the model into pickle file
file = 'WineQualityModel1.pkl'
#save file
save = pickle.dump(rf,open(file,'wb'))