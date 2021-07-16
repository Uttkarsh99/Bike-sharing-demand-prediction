#-----------------------------------------------------------------------------
#                         Step 1, Import the libraries
#-----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
#                         Step 2 - Read the CSV file
#-----------------------------------------------------------------------------

bikes = pd.read_csv('hour.csv')

#-----------------------------------------------------------------------------
#               Step 3 - Prelim Analysis and Feature Selection 
#-----------------------------------------------------------------------------

bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index', 'date', 'casual', 'registered'], axis = 1)

#basic checks of missing values
bikes_prep.isnull().sum()

# Visualise the data using pandas histogram
bikes_prep.hist(rwidth=0.9)
plt.tight_layout()

#-----------------------------------------------------------------------------
#                       Step 4 - Data Visualization
#-----------------------------------------------------------------------------

# plot the continuous features vs demand

plt.subplot(2,2,1)
plt.title("Temperature vs Demand")
plt.scatter(bikes_prep['temp'], bikes_prep['demand'], s=1, c='g')

plt.subplot(2,2,2)
plt.title("atemp vs Demand")
plt.scatter(bikes_prep['atemp'], bikes_prep['demand'], s=1, c='b')


plt.subplot(2,2,3)
plt.title("Humidity vs Demand")
plt.scatter(bikes_prep['humidity'], bikes_prep['demand'], s=1, c='m')


plt.subplot(2,2,4)
plt.title("Windspeed vs Demand")
plt.scatter(bikes_prep['windspeed'], bikes_prep['demand'], s=1, c='c')

plt.tight_layout()
# Conclusion from visualization of continuous features
#   - Predicted variable 'demand' is not normally distributed
# 	- Temperature and demand appears to have direct correlation
#	- The plot for temp and atemp appear almost identical
#	- Humidity and windspeed affect demand but need more statistical analysis

# Plot the categorical features vs demand
# Create a 3x3 subplot

plt.subplot(3,3,1)
plt.title('Average Demand per season')
# Create a list of unique season's values
cat_list = bikes_prep['season'].unique()
# Create average demand per season using groupby
cat_average = bikes_prep.groupby('season').mean()['demand']
colours = ['g', 'r', 'm', 'b']
plt.bar(cat_list, cat_average, color=colours)

# Similarly for month, holiday, weekday, year, hour, working day, weather

plt.subplot(3,3,2)
plt.title('Average Demand per month')
cat_list = bikes_prep['month'].unique()
cat_average = bikes_prep.groupby('month').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)

plt.subplot(3,3,3)
plt.title('Average Demand per holiday')
cat_list = bikes_prep['holiday'].unique()
cat_average = bikes_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)

plt.subplot(3,3,4)
plt.title('Average Demand per weekday')
cat_list = bikes_prep['weekday'].unique()
cat_average = bikes_prep.groupby('weekday').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)

plt.subplot(3,3,5)
plt.title('Average Demand per year')
cat_list = bikes_prep['year'].unique()
cat_average = bikes_prep.groupby('year').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)

plt.subplot(3,3,6)
plt.title('Average Demand per hour')
cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)

plt.subplot(3,3,7)
plt.title('Average Demand per workingday')
cat_list = bikes_prep['workingday'].unique()
cat_average = bikes_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)

plt.subplot(3,3,8)
plt.title('Average Demand per weather')
cat_list = bikes_prep['weather'].unique()
cat_average = bikes_prep.groupby('weather').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)

plt.tight_layout()

# Conclusion from visualization of categorical features
#  - There is variation in demand based on
#		- Season
#		- Month
#		- Holiday
#		- Hour
#		- Weather
#
#	- No significant change in demand due to weekday or working day
#	- Year-wise growth pattern not considered due to limited number of years
#
# Features to drop
#	- Weekdays
#	- Year
#	- Working day

#-----------------------------------------------------------------------------
#                       Step 5 - Check for Outliers
#-----------------------------------------------------------------------------
bikes_prep['demand'].describe()
bikes_prep['demand'].quantile([0.05,0.1,0.15,0.90,0.95,0.99])

#-----------------------------------------------------------------------------
#            Step 6 - Check for multiple linear regression assumptions
#-----------------------------------------------------------------------------

# 1. Linearity using correlation coefficient matrix using corr
correlation = bikes_prep[['temp','atemp','humidity','windspeed','demand']].corr()

# Conclusion After observing the correlation values
# temp and atemp are highly correlated
# Humidity and Windspeed are negatively correlated (Inverse relation) which can create multicollinearity
# Windspeed and demand are not correlated
# Therefore, Features to drop
#                - atemp
#                - Windspeed

bikes_prep = bikes_prep.drop(['weekday','year','workingday','atemp','windspeed'], axis=1)

# 2. Check the autocorrelation in 'demand' but first make sure that the values are in Float
# As the demand depends on 'hour' then it is a time series data and in this case demand of current hour is dependent on the previous hour
df1 = pd.to_numeric(bikes_prep['demand'], downcast = 'float')
plt.acorr(df1, maxlags=12) # as we are looking for 24 hours)
plt.show()
# Conclusion - There is very high autocorrelation in the 'Demand' feature

#-----------------------------------------------------------------------------
#                   Step 7 - Create/modify the variables
#                      Solving the problem of normality
#-----------------------------------------------------------------------------
# As the predicted variable 'demand' is not normally disrtibuted, we need to 
# apply some transformation

# predicted variable 'demand' is log-normal distributed
df1 = bikes_prep['demand']
df2 = np.log(df1)

plt.figure()
df1.hist(rwidth=0.9, bins = 20)
# log normal distributed

plt.figure()
df2.hist(rwidth=0.9, bins = 20)
# Normally dsitrubuted with negative skewness

#-----------------------------------------------------------------------------
#                Solving the problem of autocorrelation

# Autocorrelation in the demand column
# After seeing the autocorrelation plot, top 3 lags are considered to be very high
bikes_prep['demand'] = np.log(bikes_prep['demand'])
t_1 = df2.shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = df2.shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = df2.shift(+3).to_frame()
t_3.columns = ['t-3']

bikes_prep_lag = pd.concat([bikes_prep, t_1, t_2, t_3,], axis=1)
bikes_prep_lag = bikes_prep_lag.dropna()

#-----------------------------------------------------------------------------
#           Step 8 - Create the dummy variables and drop first to avoid 
#                          dummy variable trap using get dummies
#-----------------------------------------------------------------------------
# - season, holiday, weather, month, hour
#
bikes_prep_lag.dtypes
# Convert season, holiday, weather, month, hour to category
bikes_prep_lag['season'] = bikes_prep_lag['season'].astype('category')
bikes_prep_lag['holiday'] = bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather'] = bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['month'] = bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour'] = bikes_prep_lag['hour'].astype('category')    

bikes_prep_lag = pd.get_dummies(bikes_prep_lag, drop_first=True)

#-----------------------------------------------------------------------------
#                   Step 9 - Create Test and Train split
#-----------------------------------------------------------------------------

# Demand is a time-series dataframe
# Split the X and Y dataset in Training and testing set
Y = bikes_prep_lag[['demand']]
X = bikes_prep_lag.drop(['demand'], axis=1)

# Create the size for 70% of data
tr_size = 0.7*len(X)
tr_size = int(tr_size)

# Create train and test usind tr_size
X_train = X.values[0:tr_size]
X_test = X.values[tr_size : len(X)]

Y_train = Y.values[0:tr_size]
Y_test = Y.values[tr_size:len(Y)]

#-----------------------------------------------------------------------------
#              Step 10 - Create the model. Fit and score the model
#-----------------------------------------------------------------------------
# Linear regression
from sklearn.linear_model import LinearRegression
std_reg = LinearRegression()
std_reg.fit(X_train, Y_train)

r2_train = std_reg.score(X_train, Y_train)
r2_test = std_reg.score(X_test, Y_test) 

# Create Y predictions
Y_predict = std_reg.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(Y_test, Y_predict))

# Calculating the RMSLE for kaggle competitions
#-----------------------------------------------------------------------------
#             Final step - Calculate RMSLE and compare results
#-----------------------------------------------------------------------------
# values of Y and X are in log so converitng it back to normal by using exponential

Y_test_e = []
Y_predict_e = []

for i in range(0,len(Y_test)):
    Y_test_e.append(math.exp(Y_test[i]))
    Y_predict_e.append(math.exp(Y_predict[i]))
    
# Do the sum of logs and squares
log_sq_sum = 0.0

for i in range(0,len(Y_test_e)):
    log_a = math.log(Y_test_e[i] + 1)
    log_p = math.log(Y_predict_e[i] + 1)
    log_diff = (log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff
    
rmsle = math.sqrt(log_sq_sum/len(Y_test))
print(rmsle)
















