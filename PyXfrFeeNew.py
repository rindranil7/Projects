#=============================== Importing packages ===========================
import pandas as pd
import os
import matplotlib.pyplot as pp
import sklearn.linear_model as sklm
import sklearn.model_selection as skms
#============================== Setting up values =============================
#regCoef = [i/1000 for i in range(0, 40, 1)]
#================================ Data loading ================================
# Define path. Change it accordingly
data_path="C:\\Users\\Indranil\\Desktop\\M.Tech\\SET Project"
data_file= "DataNew.csv"
data_file= os.path.join(data_path, data_file)
# Load data from a csv file.
rd = pd.read_csv(data_file, header = 0)
# Features.From second to last but one
X = rd[list(rd)[1:-1]]
# Label.The last one
y = rd[list(rd)[-1]]
#================================= Splitting data =============================
X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.2,
                                                         random_state = 14)
mtr = X_train.shape[0]
mts = X_test.shape[0]
del X, y
#============= Regression on normalized data with regularization ==============
'''for i in regCoef:
    #Alpha is the regularization coefficient
    lm = sklm.Ridge(alpha = i, normalize = True)
    lm.fit(X_train, y_train)
    pp.plot(i, lm.score(X_train, y_train), marker = '.', color = 'r')
    pp.plot(i, lm.score(X_test, y_test), marker = '.', color = 'g')
# From above plot, regularization coefficient of 0.002 seems a good choice
# Now we will check the effect of records. We will be using optimum reg coef
# Comment out the part, once the patterns is observed
for i in range(60, mtr + 1, 2):
    #Alpha is the regularization coefficient
    lm = sklm.Ridge(alpha = 0.002, normalize = True)
    lm.fit(X_train.iloc[:i], y_train.iloc[:i])
    pp.plot(i, lm.score(X_train.iloc[:i], y_train.iloc[:i]), marker = '.',
            color = 'r')
    pp.plot(i, lm.score(X_test, y_test), marker = '.', color = 'g')'''
lm = sklm.Ridge(alpha = 0.002, normalize = True)
lm.fit(X_train, y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)
#=============================== Visualization ================================
fig = pp.figure()
#
axes = fig.add_subplot(1, 2, 1)
axes.plot([i for i in range(mtr)], pred_train,
           label = 'Train data - predicted value')
axes.plot([i for i in range(mtr)], y_train,
           label = 'Train data - actual value')
pp.xlabel('Player')
pp.ylabel('Transfer fee')
pp.title('Prediction on train data')
pp.legend()
#
axes = fig.add_subplot(1, 2, 2)
axes.plot([i for i in range(mts)], pred_test,
           label = 'Test data - predicted value')
axes.plot([i for i in range(mts)], y_test, label = 'Test data - actual value')
pp.xlabel('Player')
pp.ylabel('Transfer fee')
pp.title('Prediction on test data')
pp.legend()