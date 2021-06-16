import pandas as pd
import numpy as np
import joblib



order_data = pd.read_csv('data/machine_learning_challenge_order_data.csv')




def Preprocessing(data):
    data['customer_order_rank'].fillna(0, inplace = True)
    data = data[data['order_date'] >= '2015-03-01']
    
    for c in ['payment_id', 'platform_id', 'transmission_id']:
        data= pd.get_dummies(data, columns=[c])
        
    data.drop(['restaurant_id', 'city_id'], axis=1, inplace=True)
    
    data['order_date'] = pd.to_datetime(data['order_date'],format='%Y-%m-%d')

    data['year']=data['order_date'].dt.year 
    data['month']=data['order_date'].dt.month 
    
    data['dayofweek_num']=data['order_date'].dt.dayofweek  
    data['dayofweek_name']=data['order_date'].dt.day_name()
    
    data['weekend'] = np.where(data['dayofweek_name'].isin(['Sunday','Saturday']),1,0)
    
    data.drop('dayofweek_name', axis=1, inplace=True)
    
    for c in ['year', 'month', 'dayofweek_num']:
        data= pd.get_dummies(data, columns=[c])

    data.drop('order_date', axis=1, inplace=True)

    aggregation = dict.fromkeys(('payment_id_1491', 'payment_id_1523', 'payment_id_1619',
       'payment_id_1779', 'payment_id_1811', 'platform_id_525',
       'platform_id_22167', 'platform_id_22263', 'platform_id_22295',
       'platform_id_29463', 'platform_id_29495', 'platform_id_29751',
       'platform_id_29815', 'platform_id_30135', 'platform_id_30199',
       'platform_id_30231', 'platform_id_30359', 'platform_id_30391',
       'platform_id_30423', 'transmission_id_212', 'transmission_id_1988',
       'transmission_id_2020', 'transmission_id_4196', 'transmission_id_4228',
       'transmission_id_4260', 'transmission_id_4324', 'transmission_id_4356',
       'transmission_id_4996', 'transmission_id_21124', 'weekend', 'year_2015',
       'year_2016', 'year_2017', 'month_1', 'month_2', 'month_3', 'month_4',
       'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10',
       'month_11', 'month_12', 'dayofweek_num_0', 'dayofweek_num_1',
       'dayofweek_num_2', 'dayofweek_num_3', 'dayofweek_num_4',
       'dayofweek_num_5', 'dayofweek_num_6'), 'sum')

    aggregation2 = {'order_hour':['max', 'min'], 'customer_order_rank': ['max', 'min'],
                   'voucher_amount':['sum', 'mean'], 'delivery_fee':['sum', 'mean'], 'amount_paid':['sum', 'mean']}

    aggregation.update(aggregation2)
    data2 = data.groupby('customer_id').agg(aggregation).reset_index()
    
    data2.columns = ['_'.join(col) for col in data2.columns.values]
    data2.rename(columns = {list(data2)[0]: 'customer_id'}, inplace = True)
    
    return data2



def model_score(data):
    cols = ['amount_paid_sum', 'amount_paid_mean', 'year_2017_sum',
       'order_hour_min', 'order_hour_max', 'year_2016_sum', 'weekend_sum',
       'payment_id_1619_sum', 'transmission_id_4356_sum', 'delivery_fee_mean',
       'month_2_sum', 'delivery_fee_sum', 'transmission_id_4228_sum',
       'month_1_sum', 'dayofweek_num_5_sum', 'dayofweek_num_4_sum',
       'transmission_id_4324_sum', 'payment_id_1779_sum',
       'platform_id_29463_sum', 'dayofweek_num_0_sum', 'month_12_sum',
       'dayofweek_num_2_sum', 'platform_id_30231_sum', 'platform_id_29815_sum',
       'platform_id_30359_sum', 'year_2015_sum', 'dayofweek_num_3_sum',
       'month_11_sum', 'dayofweek_num_1_sum', 'month_10_sum', 'month_9_sum',
       'month_7_sum', 'month_5_sum', 'month_6_sum', 'month_4_sum',
       'voucher_amount_mean', 'voucher_amount_sum']
    
    data2 =data[cols]
    
    model_obj = joblib.load('model/GBM_Classifer.joblib')
   
    prediction = pd.DataFrame(columns = ['class', 'Prob_0', 'Prob_1'])
    
    prediction['class'] = model_obj.predict(data2)
    
    prediction['Prob_0'] = model_obj.predict_proba(data2)
    
    prediction['Prob_1'] = model_obj.predict_proba(data2)
  
    return prediction



def test_function():
    
    processed_data= Preprocessing(order_data)
    predictions = model_score(processed_data)
    
    assert sum(predictions.Prob_0 < 0) ==0
    assert sum(predictions.Prob_1 < 0) ==0
    assert sum(predictions.Prob_0 > 1) ==0
    assert sum(predictions.Prob_1 > 1) ==0
    assert sum((predictions['class'] == 0) | (predictions['class']== 1)) == predictions.shape[0]    
    
