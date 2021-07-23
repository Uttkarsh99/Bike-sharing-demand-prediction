# Bike-demand-share-prediction
Prediction of Bike Sharing Demand using python
- Kaggle: https://www.kaggle.com/c/bike-sharing-demand
- Dataset: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

## General Overview
---------------------------------------------------------------------------------------------------
- Predicted the demand feature using the multiple linear regression model.
- Removed irrelevant features using Exploratory Data Analysis and Correlation matrix.
- Solved the problem of Auto-correlation among the data points. Considered top 3 correlations.
- Solved the problem of Non-Normality of demand feature. Demand was log-normally distributed.
- Successfully calculated the RMSLE score of 0.356
----------------------------------------------------------------------------------------------------

## Features of the dataset
1. date
2. season - (1:winter, 2:spring, 3:summer, 4:fall)
3. year
4. month - (1:12)
5. hour
6. holiday - 1: Yes, 0: No
7. weekday - 0-6 (Sunday to Saturday)
8. workingday - 1: Yes, 0: No
9. weather - 1: Clear, 2: Mist, 3: Light rain/Light Snow, 4: Heavy rain + Ice pallets 
10. temp - Normalized temperature in celsius
11. atemp - Normalized feeling temperature in celsius
12. humidity
13. windspeed
14. casual
15. registered
16. demand

## Steps performed during the project
- Step 1 - Import the libraries
- Step 2 - Read the CSV file
- Step 3 - Prelim Analysis and Feature Selection
- Step 4 - Data Visualization
- Step 5 - Check for Outliers
- Step 6 - Check for multiple linear regression assumptions
- Step 7 - Create/modify the variables and solving the problem of normality
- Step 8 - Solving the problem of autocorrelation
- Step 9 - Create the dummy variables and drop first to avoid dummy variable trap using get dummies
- Step 9 - Create Test and Train split
- Step 10 - Create the model. Fit and score the model
- Final step - Calculate RMSLE and compare results




### Graph of demand vs categorical features
![image](https://user-images.githubusercontent.com/63557791/126624967-551099c6-8a66-415e-82d1-847defd6dca6.png)

**Data visualization Analysis results of Categorical Features**
<ol> 
<li>There is variation in demand based on</li>
<ol>
<li>Season - Highest demand in Fall season and Lowest demand in Spring season </li>
<li>Month - High demand from May to October </li>
<li>Holiday - Demand is less on holidays</li>
<li>Hour - Peak demand at 8am and 5pm</li>
<li>Weather - Highest demand in clear weather and Lowest demand in heavy rainy weather </li>
</ol>
<li>No significant change in demand due to weekday or working day</li>
<li>Year-wise growth pattern not considered due to limited number of years</li>
</ol>

**Features to drop**
<ol>
<li>Weekdays</li>
<li>Year</li>
<li>Working day</li>
</ol>

### Graph of demand vs continuous features
![image](https://user-images.githubusercontent.com/63557791/126641645-21734eed-a4c5-4270-878a-0a8ce781247d.png)

### Results after doing EDA
Data visualization Analysis results of Continuous Features

- Predicted variable 'demand' is not normally distributed
- Temperature and demand appears to have direct correlation
- The plot for temp and atemp appear almost identical
- Humidity and windspeed need more statistical analysis

**Features to drop**
<ol>
<li> atemp </li>
<li> windspeed </li>
</ol>

### Log noarmally distributed demand feature
![image](https://user-images.githubusercontent.com/63557791/126625621-fc19af31-79e5-40cf-bac0-7e70e079635e.png)

### After transformation: Normally distributed demand feature with negative skewness
![image](https://user-images.githubusercontent.com/63557791/126625800-4016f3ad-5b4f-4ce1-a4a6-21030c846627.png)
