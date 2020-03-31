import numpy as np

train_data_columns = ['Period', 'EQ', 'Social_Search_Impressions', 'Social_Search_Working_cost', 'Digital_Impressions', 'Digital_Working_cost', 'Print_Impressions.Ads40', 'Print_Working_Cost.Ads50', 'OOH_Impressions', 'OOH_Working_Cost', 'SOS_pct', 'Digital_Impressions_pct', 'CCFOT', 'Median_Temp', 'Median_Rainfall', 'Fuel_Price', 'Inflation', 'Trade_Invest', 'Brand_Equity', 'Avg_EQ_Price', 'Any_Promo_pct_ACV', 'Any_Feat_pct_ACV', 'Any_Disp_pct_ACV', 'EQ_Base_Price', 'Est_ACV_Selling', 'pct_ACV', 'Avg_no_of_Items', 'pct_PromoMarketDollars_Category', 'RPI_Category', 'Magazine_Impressions_pct', 'TV_GRP', 'Competitor1_RPI', 'Competitor2_RPI', 'Competitor3_RPI', 'Competitor4_RPI', 'EQ_Category', 'EQ_Subcategory', 'pct_PromoMarketDollars_Subcategory', 'RPI_Subcategory']

def percentage_to_float(x):
    '''convert variable percent to float'''
    return x/100

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100