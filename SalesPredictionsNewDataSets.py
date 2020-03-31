import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10.0, 8.0)
pd.set_option('display.max_columns',40)
from Skillenz_Hackthon.SalesPredictionUtils import *

################################ load data #######################

train_data = pd.read_excel("C:\\Users\\NAGENDRA_CHAPALA\\Documents\\7400618_810833986\\DataScienceCourse_TechEssential\\course\\DATA_SCIENCE_AUTHORITY\\Skillenz_Hackthon\\New_Data_Sets\\TrainingData.xlsx")
train_data.shape
train_data.head()
train_data.columns

train_data.describe()
train_data.info()

test_data = pd.read_excel("C:\\Users\\NAGENDRA_CHAPALA\\Documents\\7400618_810833986\\DataScienceCourse_TechEssential\\course\\DATA_SCIENCE_AUTHORITY\\Skillenz_Hackthon\\New_Data_Sets\\TestData.xlsx")
test_data.shape
test_data.columns

test_data.describe()
test_data.info()

print('The train data has {0} rows and {1} columns'.format(train_data.shape[0],train_data.shape[1]))
print('----------------------------')
print('The test data has {0} rows and {1} columns'.format(test_data.shape[0],test_data.shape[1]))

############################### Data Preparation ####################################

#check missing values
train_data.columns[train_data.isnull().any()]
test_data.columns[test_data.isnull().any()]

#missed data
miss_sum = train_data.isnull().sum()
miss_sum = miss_sum[miss_sum > 0]
miss_sum.sort_values(inplace=True)
miss_sum
type(miss_sum)

#visualising missing values
miss_sum = miss_sum.to_frame()
miss_sum.columns = ['count']
miss_sum
miss_sum.index.names = ['Name']
miss_sum['Name'] = miss_sum.index

#plot the missing value count
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=miss_sum)
plt.xticks(rotation = 90)
plt.show()

#change Period variable
train_data['Period'] = train_data['Period'].str.slice_replace(4, 14, '')
test_data['Period'] = test_data['Period'].str.slice_replace(4, 14, '')

#Impute missing values
from missingpy import MissForest
imputer = MissForest()
train_data_imputed = imputer.fit_transform(train_data)

train_data_imputed = pd.DataFrame(data=train_data_imputed[0:,0:],
index=[i for i in range(train_data_imputed.shape[0])],
columns= train_data_columns)

train_data_imputed.columns

#train_data_imputed.reset_index(drop=True).reset_index(drop=True)

type(train_data_imputed)
train_data_imputed.head(10)
# write csv
train_data_imputed.to_excel('train_data_imputed.xlsx', index=False)


#check missing values after imputation
train_data_imputed.columns[train_data_imputed.isnull().any()]
train_data_imputed.index

#convert percentage columns to float
percentage_cols = ['Social_Search_Impressions','Digital_Impressions','Print_Impressions.Ads40','OOH_Impressions','Digital_Impressions_pct','Any_Promo_pct_ACV','Any_Feat_pct_ACV','Any_Disp_pct_ACV','Magazine_Impressions_pct']
for col in percentage_cols:
    train_data_imputed[col] = train_data_imputed[col].apply(percentage_to_float)

train_data.head(5)
del train_data_imputed['Period']

#correlation plot
corr = train_data_imputed.corr()
corr
type(corr)
sns.heatmap(corr)
corr.columns
corr['EQ']

print (corr['EQ'].sort_values(ascending=False)[:15], '\n') #top 10 values
print ('----------------------')
print (corr['EQ'].sort_values(ascending=False)[-5:]) #last 5 values

############################# data visualization ########################

high_corr_df = train_data[['EQ','Median_Rainfall','Social_Search_Impressions','pct_PromoMarketDollars_Category','Inflation','EQ_Category','pct_PromoMarketDollars_Subcategory','EQ_Subcategory','pct_ACV','OOH_Working_Cost','Median_Temp','RPI_Category','EQ_Base_Price','Any_Feat_pct_ACV','Digital_Impressions_pct']]

fig, axes = plt.subplots(nrows=4, ncols=4)
for i, column in enumerate(train_data.columns):
    sns.distplot(train_data[column],ax=axes[i//4,i%4])

high_corr_df[['EQ','Median_Rainfall','Social_Search_Impressions']].head(2)
high_corr_df[['EQ','Median_Rainfall','Social_Search_Impressions']].boxplot()

sns.distplot(high_corr_df['Social_Search_Impressions'])

#p = pd.melt(train_data, id_vars='EQ', value_vars=train_data.columns)
#sns.boxplot(x="value", y="EQ", data=pd.melt(high_corr_df, id_vars='EQ'))

############################# feature engineering ##################################

#transform the numeric features using log(x + 1)
from scipy.stats import skew
skewed = train_data_imputed.apply(lambda x: skew(x.dropna().astype(float)))
skewed
positive_skewed = skewed[skewed > 0.75]
positive_skewed
skewed = skewed.index
train_data_imputed[skewed] = np.log1p(train_data_imputed[skewed])
negative_skewed = skewed[skewed < -0.5]
negative_skewed

test_data_without_period = test_data.drop('Period', axis=1)
skewed_test_data = test_data_without_period.apply(lambda x: skew(x.dropna().astype(float)))
skewed_test_data
test_positive_skewed = skewed[skewed > 0.75]
test_positive_skewed
test_negative_skewed = skewed[skewed < -0.5 ]
test_negative_skewed



#distibute target variable
sns.distplot(train_data['EQ'])
#skewness of target variable
print("The skewness of Sales variable is {}".format(train_data['EQ'].skew()))
print("The kurtosis of Sales variable is {}".format(train_data['EQ'].kurtosis()))

#create a label set
label_df = pd.DataFrame(index = train_data_imputed.index, columns = ['EQ'])
label_df['EQ'] = train_data['EQ']
#label_df['EQ'] = np.log(train_data['EQ'])
label_df.shape

#del train_data['EQ']
#train_data.shape
#train_data.columns
######################## Divide data into train and test data ########################
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
x_train , x_test , y_train , y_test = train_test_split(train_data_imputed, label_df, test_size = 0.20, random_state = 2)

#take actual validation data
validation_test_data = test_data.loc[:, test_data.columns != 'EQ']
validation_test_data.shape

#take validation label data
validation_label_df = pd.DataFrame(index = test_data.index, columns = ['EQ'])
validation_label_df['EQ'] = test_data['EQ']
validation_label_df.shape

################################## STATSMODELS ###########################
# create a fitted model with all three features
#+ Print_Impressions.Ads40 + Print_Working_Cost.Ads50
import statsmodels.formula.api as smf
lm1 = smf.ols(formula='EQ ~ Social_Search_Impressions + Social_Search_Working_cost + Digital_Impressions + Digital_Working_cost + OOH_Impressions + OOH_Working_Cost + SOS_pct + Digital_Impressions_pct + CCFOT + Median_Temp + Median_Rainfall + Fuel_Price + Inflation + Trade_Invest + Brand_Equity + Avg_EQ_Price + Any_Promo_pct_ACV + Any_Feat_pct_ACV + Any_Disp_pct_ACV + EQ_Base_Price + Est_ACV_Selling + pct_ACV + Avg_no_of_Items + pct_PromoMarketDollars_Category + RPI_Category + Magazine_Impressions_pct + TV_GRP + Competitor1_RPI + Competitor2_RPI + Competitor3_RPI + Competitor4_RPI + EQ_Category + EQ_Subcategory + pct_PromoMarketDollars_Subcategory + RPI_Subcategory', data=train_data).fit()
# print the coefficients
lm1.summary()

lm2 = smf.ols(formula='EQ ~ Social_Search_Impressions + OOH_Working_Cost + Median_Rainfall + Inflation + pct_PromoMarketDollars_Category + RPI_Category + Magazine_Impressions_pct + EQ_Category + EQ_Subcategory + pct_PromoMarketDollars_Subcategory', data=train_data).fit()
# print the coefficients
lm2.summary()

#need check multicollinarity and verifg again
x_test.shape
y_test.shape
x_test.columns[x_test.isnull().any()]
####################### LinearRegression ##############################################
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
# calculate MAE, MSE, RMSE
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse))
reg.score(x_test,y_test)

#calculate Mean absolute percentage error (MAPE)
from Skillenz_Hackthon.SalesPredictionUtils import *
print(mean_absolute_percentage_error(y_test, y_pred))

df_y_pred = pd.DataFrame(y_pred,index=y_pred[:,0], columns=['EQ'])
type(df_y_pred)
df_y_pred.columns
df_y_pred.shape
df_y_pred.info()
df_y_pred.head(10)

df_y_pred['EQ'] = df_y_pred['EQ'].astype(float)
y_test['EQ'] = y_test['EQ'].astype(float)
df_y_pred.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)
#dfmas['ma5x'] = ma5xdf['ma5x']

y_test_new = y_test.rename(columns = {'EQ':'Actual'})
y_pred_new = df_y_pred.rename(columns = {'EQ':'Predicted'})
df_col = pd.concat([y_test_new, y_pred_new], axis=1)
df_col.head(20)

############# checking with validation data #########################

y_pred_validation = reg.predict(validation_test_data)
# calculate MAE, MSE, RMSE
print(mean_absolute_error(validation_label_df, y_pred_validation))
print(mean_squared_error(validation_label_df, y_pred_validation))
rmse_validation = np.sqrt(mean_squared_error(validation_label_df, y_pred_validation))
print("RMSE: %f" % (rmse_validation))
reg.score(validation_label_df, y_pred_validation)

######################### Lasso & Ridge####################################
from sklearn.linear_model import Lasso
#found this best alpha through cross-validation
best_alpha = 0.00099
regr_lasso = Lasso(alpha=best_alpha, max_iter=50000)
regr_lasso.fit(x_train, y_train)
y_pred = regr_lasso.predict(x_test)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse_lasso))


################################ GradientBoostingRegressor ###########################
from sklearn import ensemble
gbregressor = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')

gbregressor.fit(x_train, y_train.values.ravel())
y_pred_gbreg = gbregressor.predict(x_test)
xgb_regr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gbreg))
print("gbregressor RMSE: %f" % (xgb_regr_rmse)) #gbregressor RMSE: 0.014072


# Calculate the feature ranking - Top 10
importances = gbregressor.feature_importances_
indices = np.argsort(importances)[::-1]
print("Sales Prediction - Top 10 Important Features\n")
for f in range(20):
    print("%d. %s (%f)" % (f + 1, train_data.columns[indices[f]], importances[indices[f]]))
#Plot the feature importances of the forest
indices=indices[:10]
plt.figure()
plt.title("Top 10 Feature importances")
plt.bar(range(10), importances[indices], color="r", align="center")
plt.xticks(range(10), train_data.columns[indices], fontsize=14, rotation=45)
plt.xlim([-1, 10])
plt.show()
#Mean Feature Importance
print("Mean Feature Importance %.6f" %np.mean(importances))

gbregressor.score(x_test,y_test)

################################# XGBRegressor ########################
import xgboost as xgb
from sklearn.metrics import mean_squared_error
xgb_regr = xgb.XGBRegressor(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=6,
                       min_child_weight=1.5,
                       n_estimators=7200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=42,
                       silent=1)

xgb_regr.fit(x_train, y_train)
y_pred_xgb = xgb_regr.predict(x_test)
xgb_regr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print("xgb RMSE: %f" % (xgb_regr_rmse)) #xgb RMSE: 0.152788
xgb_regr.score(x_test, y_test) #0.9928439315132127









