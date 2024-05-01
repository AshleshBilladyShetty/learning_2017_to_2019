######################################################################################
#STEP1: Data Preparation Code
#####################################################################################

#function that can generate any lag value we want.

import pandas as pd
    
#first try the shift 
#manipulate the supervise_shift function

ridership = pd.read_csv('D:\\FT2 - Team 7\\Data\\Fall 2017 Predictive Case\\Model Buiding Script\\Usage.csv', parse_dates = [0])[['Date', 'Ridership']]


def series_to_supervised(data, lags, dropnan=False):
    """data :usage frame"""
    time_series = data['Ridership']
    cols, names = list(), list()
    #append original
    cols.append(data['Date'])
    cols.append(time_series.shift(0))
    names.append('Date')
    names.append('original')
    
    for i in lags:
        cols.append(time_series.shift(i))
        names.append('lag(t-%d)' % (i))
        
    agg = pd.concat(cols, axis = 1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace = True)
    return agg


def frame_with_multiple_lag_static(lags, school = True, weather = True):
    """create static lag frames, together with features"""
    usage = pd.read_csv('D:\\FT2 - Team 7\\Data\\Fall 2017 Predictive Case\\Model Buiding Script\\Usage.csv', parse_dates = [0])[['Date', 'Ridership']]
    usage_with_lags = series_to_supervised(usage, lags, dropnan = True)
    

    calendar = pd.read_csv('D:\\FT2 - Team 7\\Data\\Fall 2017 Predictive Case\\Model Buiding Script\\calendar_v3.csv', parse_dates = [0, 2])
    #clean up calender
    drop_calendar = ['Date.1', 'LYDate', '2YDate', 'LYHolidayDate', '2YHolidayDate',\
                 'Holiday', 'Holiday_date', 'Holiday_date_before','Holiday_date_after',\
                 'Week', 'Month','lag1Y', 'lag2Y', 'LYavg','lag1Holiday', 'lag2Holiday', 'Holidayavg', \
                 'day', 'month']
    calendar_feature = calendar.drop(drop_calendar, axis = 1)
    usage_new = pd.merge(usage_with_lags, calendar_feature, on = 'Date')
    
    #cuz we need to use weather and school from last year and therefore the LY Date
    dates = calendar[['Date', 'LYDate']]
    usage_new = pd.merge(usage_new, dates, on = 'Date')
    #append weather
    if weather:
        weather_feature = pd.read_csv('D:\\FT2 - Team 7\\Data\\Fall 2017 Predictive Case\\Model Buiding Script\\Weather.csv', parse_dates = [0])
        usage_new = pd.merge(usage_new, weather_feature, left_on = 'LYDate', right_on = 'Date')
        usage_new = usage_new.rename(columns = {'Date_x':'Date'})
        usage_new = usage_new.drop('Date_y', axis = 1)
    #append school
    if school:
        school_feature = pd.read_csv('file:///D:/FT2 - Team 7/Data/Fall 2017 Predictive Case/Cleaned data/school_calender_v2.csv',parse_dates = [1], index_col = [0])
        drop_school = ['DayWeek', 'Off']
        school_feature = school_feature.drop(drop_school, axis = 1)
        usage_new = pd.merge(usage_new, school_feature, on = 'Date')
        #usage_new = usage_new.rename(columns = {'Date_x':'Date'})
        #usage_new = usage_new.drop('Date_y', axis = 1)
    
    usage_new = usage_new.drop('LYDate', axis = 1)
    return usage_new

def frame_with_multiple_lag_dynamic(lags, school = True, weather = True):
    """create dynamic lag frames, together with features"""
    usage = pd.read_csv('Usage.csv', parse_dates = [0])[['Date', 'Ridership']]
    usage_with_lags = series_to_supervised(usage, lags, dropnan = True)
    

    calendar = pd.read_csv('calendar_v3.csv', parse_dates = [0, 2])
    #clean up calender
    drop_calendar = ['Date.1', 'LYDate', '2YDate', 'LYHolidayDate', '2YHolidayDate',\
                 'Holiday', 'Holiday_date', 'Holiday_date_before','Holiday_date_after',\
                 'Week', 'Month','lag1Y', 'lag2Y', 'LYavg','lag1Holiday', 'lag2Holiday', 'Holidayavg',\
                 'day', 'month']
    calendar_feature = calendar.drop(drop_calendar, axis = 1)
    usage_new = pd.merge(usage_with_lags, calendar_feature, on = 'Date')
   
    #append weather
    if weather:
        weather_feature = pd.read_csv('Weather.csv', parse_dates = [0])
        usage_new = pd.merge(usage_new, weather_feature, on = 'Date')
    #append school
    if school:
        school_feature = pd.read_csv('file:///D:/FT2 - Team 7/Data/Fall 2017 Predictive Case/Cleaned data/school_calender_v2.csv',parse_dates = [1], index_col = [0])
        drop_school = ['DayWeek', 'Off']
        school_feature = school_feature.drop(drop_school, axis = 1)
        usage_new = pd.merge(usage_new, school_feature, on = 'Date')
    return usage_new
    

def frame_with_multiple_LYlag_static(lags, school = True, weather = True):
    """create static lag frames based on LYDate, together with features"""
    usage = pd.read_csv('Usage.csv', parse_dates = [0])[['Date', 'Ridership']]
    usage_with_lags = series_to_supervised(usage, lags, dropnan = True)
    

    calendar = pd.read_csv('calendar_v3.csv', parse_dates = [0, 2])
    #clean up calender
    drop_calendar = ['Date.1', '2YDate', 'LYHolidayDate', '2YHolidayDate',\
                 'Holiday', 'Holiday_date', 'Holiday_date_before','Holiday_date_after',\
                 'Week', 'Month','lag1Y', 'lag2Y', 'LYavg','lag1Holiday', 'lag2Holiday', 'Holidayavg',\
                 'day', 'month']
    calendar_feature = calendar.drop(drop_calendar, axis = 1)
    usage_new=pd.merge(usage, calendar_feature, on = 'Date')
    usage_new = pd.merge(usage_with_lags, usage_new, left_on = 'Date', right_on = 'LYDate')
    usage_new = usage_new.rename(columns = {'Date_y':'Date'})
    usage_new = usage_new.drop('Date_x', axis = 1)
    #cuz we need to use weather and school from last year and therefore the LY Date
    #append weather
    if weather:
        weather_feature = pd.read_csv('Weather.csv', parse_dates = [0])
        usage_new = pd.merge(usage_new, weather_feature, left_on = 'LYDate', right_on = 'Date')
        usage_new = usage_new.rename(columns = {'Date_x':'Date'})
        usage_new = usage_new.drop('Date_y', axis = 1)
    #append school
    if school:
        school_feature = pd.read_csv('file:///D:/FT2 - Team 7/Data/Fall 2017 Predictive Case/Cleaned data/school_calender_v2.csv',parse_dates = [1], index_col = [0])
        drop_school = ['DayWeek', 'Off']
        school_feature = school_feature.drop(drop_school, axis = 1)
        usage_new = pd.merge(usage_new, school_feature, left_on = 'LYDate', right_on = 'Date')
        usage_new = usage_new.rename(columns = {'Date_x':'Date'})
        usage_new = usage_new.drop('Date_y', axis = 1)
    
    usage_new = usage_new.drop('LYDate', axis = 1)

    return usage_new

######################################################################################
#Step2: Model Evaluation code 
#####################################################################################

import numpy as np
np.set_printoptions(suppress=True)

class predTrt_evalMet:    
    def __init__(self, y_pred, y_actual):
        self.y_pred = np.array(y_pred)
        self.y_actual = np.array(y_actual)
       
    def predTrt(self):
        act = self.y_actual
        pred = self.y_pred
        i = np.where(act == 0)
        pred[i] = 0
        return pred
 
    def evalMet(self):
        act = self.y_actual
        pred = self.y_pred
        i = np.where(act == 0)
        act[i]= pred[i]
        error = act - pred                  
        RMSE = np.sqrt(np.mean(error**2))
        MAE = np.mean(np.absolute(error))     
        MAPE = np.mean(np.absolute(error)/(act * 1.0))
        return print(' RMSE = {:.2f} \n MAE = {:.2f} \n MAPE = {:.00%}'.format(RMSE,MAE,MAPE))

##################################################################################################
#Step 3 Modeling Code
###################################################################################################

########### Step3.1 : Preprocessing of the data for building models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance, DMatrix, plot_tree


from SophieDataPrepCode import frame_with_multiple_lag_static, frame_with_multiple_LYlag_static

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn import preprocessing

lags_o = list(range(363, 413, 7))
lags_o += list(range(728, 743, 7))
lags_o += list(range(364, 414, 7))
#lags_o = list(range(363, 750))


frame_initial = frame_with_multiple_lag_static(lags = lags_o)

#try without flag
drop_columns = ['Columbus Day', 'Easter', 'Eid Al-Fitr', 'Eid Al-Fitr - Day After',
       'Eid al-Adha', 'Eid al-Adha - Day After', 'Fourth of July',
       'Good Friday', 'Halloween', 'Labor Day', 'MEA Friday', 'MEA Saturday',
       'MEA Thursday', 'MLK Day', 'Memorial Day', 'Mpls School Patrol',
       "New Year's Day", "New Year's Eve", "Palm Sunday", "President's Day",
       "St. Patrick's Day", "Suburban School Patrol Day 1",
       'Suburban School Patrol Day 2']

frame_new = frame_initial.drop(drop_columns, axis = 1)


#calendar = pd.read_csv('D:\\FT2 - Team 7\\Data\\Fall 2017 Predictive Case\\Model Buiding Script\\calendar_v2.csv', parse_dates = [0])
#calendar = calendar[['Date','LYavg', 'Holidayavg']]
#frame_boost = pd.merge(frame_new, calendar, on = 'Date')
#frame_boost['Avg_normal'] = np.where(frame_new['Flag_holiday'] == 1, 0, frame_boost['LYavg'])
#frame_boost = frame_boost.rename(columns = {'Holidayavg':'Avg_holiday'})
#frame_boost = frame_boost.fillna(0)
#frame_boost = frame_boost.drop('LYavg', axis = 1)
##frame_boost = frame_boost.drop('avg_m', axis = 1)
#moving_avg =  pd.read_csv("D:\\FT2 - Team 7\\Data\\Fall 2017 Predictive Case\\ly_week_month.csv",parse_dates=[0])
#frame_boost = pd.merge(frame_boost, moving_avg, on  = 'Date')



train_data = pd.read_csv('TrainingData_24Nov2017.csv',parse_dates = ['Date'], infer_datetime_format = True)
test_data = pd.read_csv('TestingData_24Nov2017.csv',parse_dates = ['Date'], infer_datetime_format = True)

frame_new1 = frame_new.drop(['original'],1)

td = train_data.merge(frame_new1, how = 'left', on =['Date'])
tsd = test_data.merge(frame_new1, how = 'left', on =['Date'])

trnd = td.drop(['Date','dayNew','yearNew','weekday','AverageofRidership_V','NickHoursOpenAdj_V','AvgLTemp_V','AvgSnw_V'], axis =1)
tesd = tsd.drop(['Date','dayNew','yearNew','weekday','AverageofRidership_V','NickHoursOpenAdj_V','AvgLTemp_V','AvgSnw_V'], axis =1)

x_train = trnd.iloc[:,1:]
X_train_lag = pd.DataFrame(preprocessing.scale(x_train), columns = x_train.columns)
y_train_lag = trnd.iloc[:,0]

x_test = tesd.iloc[:,1:]
X_test_lag = pd.DataFrame(preprocessing.scale(x_test), columns = x_test.columns)
y_test_lag = tesd.iloc[:,0]

######### 3.2: Rough First Attempt Forecasting With XGBoost: To understand naive pattern in the data before deepdiving to extensive forecasting

###############3.2.1: Model parameter tuning
import xgboost as xgb

xgb_regressor = xgb.XGBRegressor(n_estimators=500, learning_rate=0.068, gamma=0.02, subsample=0.6,
                           colsample_bytree=0.8, max_depth=10, seed = 6,reg_lambda = 0.5 )

param_grid = {'max_depth' : list(np.random.randint(2,9,1)),
              'learning_rate' : list(np.random.uniform(0.05,0.1,10)),
              'n_estimators' : list(np.random.randint(1000,1001,1)),
              'subsample':list(np.random.uniform(0.3,0.81,10)),
              'objective': ['reg:linear']
              }

model = XGBRegressor(seed = 2)
eval_set = [(X_test_lag,y_test_lag)]
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
#grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
rsearch = RandomizedSearchCV(estimator=model, 
                             param_distributions=param_grid, 
                             n_iter=100)
rsearch.fit(X_train_lag, y_train_lag, 
            early_stopping_rounds=10,
            eval_metric = 'rmse', 
            eval_set = eval_set, 
            verbose = True)

xgb_regressor.fit(X_train_lag, y_train_lag)
math.sqrt(mean_squared_error(y_test_lag,np.round(rsearch.predict(X_test_lag), -2)))

model = XGBRegressor(seed =2,**rsearch.best_params_)
model.fit(X_train_lag, y_train_lag)

plot_importance(model)
plt.show()
model


######

reg = XGBRegressor(seed = 2)
reg_pram = {'n_estimators':[300,400,500],'learning_rate':[0.01, 0.05,0.1],'subsample':[0.4,0.6,0.8]}
reg_search = GridSearchCV(reg, reg_pram, cv = 5, scoring='neg_mean_squared_error')
reg_search.fit(x_train, y_train)


########

trnd = td.drop(['Date','dayNew','yearNew','weekday','AverageofRidership_V','NickHoursOpenAdj_V','AvgLTemp_V','AvgSnw_V','AvgRidershipHoliAdj_V'], axis =1)
tesd = tsd.drop(['Date','dayNew','yearNew','weekday','AverageofRidership_V','NickHoursOpenAdj_V','AvgLTemp_V','AvgSnw_V','AvgRidershipHoliAdj_V'], axis =1)

x_train = trnd.iloc[:,1:]
X_train_lag = pd.DataFrame(preprocessing.scale(x_train), columns = x_train.columns)
y_train_lag = trnd.iloc[:,0]

x_test = tesd.iloc[:,1:]
X_test_lag = pd.DataFrame(preprocessing.scale(x_test), columns = x_test.columns)
y_test_lag = tesd.iloc[:,0]

import xgboost as xgb

xgb_regressor = xgb.XGBRegressor(n_estimators=500, learning_rate=0.068, gamma=0.02, subsample=0.6,
                           colsample_bytree=0.8, max_depth=10, seed = 6,reg_lambda = 0.5 )

param_grid = {'max_depth' : list(np.random.randint(2,9,1)),
              'learning_rate' : list(np.random.uniform(0.05,0.1,10)),
              'n_estimators' : list(np.random.randint(1000,1001,1)),
              'subsample':list(np.random.uniform(0.3,0.81,10)),
              'objective': ['reg:linear']
              }

###############3.2.2: Model performance evaluation

model = XGBRegressor(seed = 2)
eval_set = [(X_test_lag,y_test_lag)]
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
#grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
rsearch = RandomizedSearchCV(estimator=model, 
                             param_distributions=param_grid, 
                             n_iter=100)
rsearch.fit(X_train_lag, y_train_lag, 
            early_stopping_rounds=10,
            eval_metric = 'rmse', 
            eval_set = eval_set, 
            verbose = True)

xgb_regressor.fit(X_train_lag, y_train_lag)
math.sqrt(mean_squared_error(y_test_lag,np.round(rsearch.predict(X_test_lag), -2)))

model = XGBRegressor(seed =2,**rsearch.best_params_)
model.fit(X_train_lag, y_train_lag)

plot_importance(model)
plt.show()
model

#########################################################################
######### 3.3: Second attempt at XGBoost using randomized search for parameter tuning which boosted the model performance
########################################################################

### XG Boost 

# Step1: Get necessary libraries
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from EvaluationCalculatorFunction import*
from xgboost import XGBRegressor, plot_importance, DMatrix, plot_tree
#from datetime import datetime
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score,make_scorer, mean_absolute_error
#from sklearn.grid_search import GridSearchCV

    ##Step2: Get the training and test datasets ready
print('{0} \n {1}'.format(list(x_train.columns) ,list(x_train.shape))); 
print('{0} \n {1}'.format(list(x_test.columns) ,list(x_test.shape)));     
print('{0}'.format(list(y_train.shape)));     
print('{0}'.format(list(y_test.shape)));     

x_train_arr = x_train
y_train_arr = y_train
x_test_arr = x_test
y_test_arr = y_test


    ##Step 4: Set hyper paramteters
param_grid = {'max_depth' : list(np.random.randint(2,9,1)),
              'learning_rate' : list(np.random.uniform(0.1,0.4,10)),
              'n_estimators' : list(np.random.randint(1000,1001,1)),
              'subsample':list(np.random.uniform(0.3,0.81,10)),
              'objective': ['reg:linear']
              }

    ##Step 5: Set the model and model parameters and model evaluation inputs
model = XGBRegressor(seed = 2)
eval_set = [(x_test_arr,y_test_arr)]
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
#grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
rsearch = RandomizedSearchCV(estimator=model, 
                             param_distributions=param_grid, 
                             n_iter=100)
rsearch.fit(x_train_arr, y_train_arr, 
            early_stopping_rounds=10,
            eval_metric = 'rmse', 
            eval_set = eval_set, 
            verbose = True)

#summarize the hyperparametertuning results
rsearch.best_score_
rsearch.best_params_
rsearch.best_estimator_ 
rsearch.cv_results_['params']

    ##Step 6: Use the best hyperparameters to fit the model on training dataset and analyse modelresults

model = XGBRegressor(seed =2,**rsearch.best_params_)
model.fit(x_train_arr, y_train_arr)

plot_importance(model)
plt.show()

#plot_tree(model)
#plt.show()

    ##Step 7: use the best parameters to score the test sample
y_pred = rsearch.predict(x_test_arr)
predictions = [round(value) for value in y_pred]

    ##Step 8: Evaluate the model performance both on train and test
pred_y_trtd = predTrt_evalMet(y_pred,y_test_arr).predTrt()
predTrt_evalMet(y_pred,y_test_arr).evalMet()

pred_y_trtd = predTrt_evalMet(y_pred,y_train_arr).predTrt()
predTrt_evalMet(y_pred,y_train_arr).evalMet()


#########################################################################
######### 3.3: LSTM based forecasting implementation
########################################################################

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,LSTM,Activation
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

from pandas import read_csv
from datetime import datetime
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
# load dataset
 
# get value
    
from Data_prep_lag_values_sophie import frame_with_multiple_lag_dynamic
#usage = pd.read_csv('D:\\FT2 - Team 7\\Data\\Fall 2017 Predictive Case\\Dynamic_data_lstm_1128.csv', index_col=[0],parse_dates=[0]) 
usage = frame_with_multiple_lag_dynamic(lags = [14, 363, 364])

drop_columns = ['Date', 'Flag_Biweekend', 'lag1Y', 'lag2Y', 'lag1Holiday',\
                'lag2Holiday', 'dayweek', 'LYavg','avg_m',\
                'Holidayavg', 'Tem_H', 'Temp_avg',\
       'Temp_L', 'DewPnt_H', 'DewPnt_L', 'Humid_H', \
       'Humid_L', 'Presr_H', 'Presr_avg', 'Presr_L', 'Visib_H', 'Visib_avg',\
       'Visib_L', 'Wind_avg', 'Wind_L', 'Wind_H', 'avg_m', 'ly_week_ridership', 'ly_month_ridership']

#keep_columns = ['original', 'avg_m']

drop_new = [ 'Past2Week_avg', 'lag(t-14)_avg',
       'lag(t-363)_avg', 'lag(t-364)_avg', 'DewPnt_avg',
       'Humid_avg', 'PrecipTotal_In', 'SnowTotal_In', 'Days_before_next_holiday',
       'Days_after_previous_holiday']

usage = usage.drop(drop_columns, axis = 1)

usage = usage.drop(drop_new, axis = 1)
usage = usage.dropna()



values = usage.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_days = 1
n_features = usage.shape[1] - 1
# frame as supervised learning
# =============================================================================
# reframed = series_to_supervised(scaled, n_days, 1)
# print(reframed.shape)
# print(reframed.head())
# =============================================================================

# split into train and test sets
# values = reframed.values
n_train_days = len(scaled) - 365
train = scaled[:n_train_days, :] #rmb to update train size to consistent with batch
test = scaled[n_train_days:, :]

# split into input and outputs
n_obs = n_days * n_features
train = scaled[:n_train_days, :] #rmb to update train size to consistent with batch
test = scaled[n_train_days:, :]

train_X, train_y = train[:, 1:], train[:, 0]
test_X, test_y = test[:, 1:], test[:, 0]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
# Define early_stopping_monitor
#early_stopping_monitor = EarlyStopping(patience=3)
model = Sequential()
model.add(LSTM(112, return_sequences = True, input_shape=(train_X.shape[1], train_X.shape[2]))) #return_sequences=True, 
model.add(Activation('tanh'))
model.add(Dropout(0.1))
#model.add(Dropout(0.3))
model.add(LSTM(112))
model.add(Dropout(0.28))
#model.add(Dense(35))
#model.add(Activation('hard_sigmoid'))
model.add(Dense(1))
model.add(Activation('hard_sigmoid'))
model.compile(loss='mse', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=42, validation_data=(test_X, test_y) ,verbose=2, shuffle=False) #validation_split=0.1, callbacks=[early_stopping_monitor],
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

###Update the order for dynamic model
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_days*(n_features)))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


###Plotting for prediction
y_test_plot = DataFrame(inv_y)
y_test_plot['y_pred'] = inv_yhat
y_test_plot.to_csv("D:\\FT2 - Team 7\\Data\\Fall 2017 Predictive Case\\LSTMdynamic_result_biweek.csv")

new = pd.concat([pd.Series(inv_y).to_frame(),pd.Series(inv_yhat)], axis = 1)
new.columns = ['y', 'lstm']
new.to_csv('dynamic_static_pre.csv')


