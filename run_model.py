import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# normalize feature
def norm_feature(df):
    from sklearn.preprocessing import normalize
    
    # normalize df feature
    norm_df = pd.DataFrame(normalize(np.array([df['uxp_reorder_ratio'], df['u_total_bought_product'], df['u_avg_bought_p_per_order'], df['uxp_order_ratio']])).T)
    
    df['uxp_reorder_ratio'] = norm_df[0]
    df['u_total_bought_product'] = norm_df[1]
    df['u_avg_bought_p_per_order'] = norm_df[2]
    df['uxp_order_ratio'] = norm_df[3]
    
    print('normalized feature')
    
    return df

# return a trained model
def train_xgb(df):
    import xgboost as xgb
    
    start_time = time.time()
    
    print('start training xgboost classifier')
    
    
    train = df[df['eval_set'] == 'train'].copy()
    
    # get train data / label
    X_train, y_train = train.drop(['eval_set', 'user_id', 'product_id', 'order_id', 'reordered'], axis=1), train.reordered
    
    # xgb paramter
    parameters = {'eval_metric':'logloss', 
                  'max_depth':'10', 
                  'colsample_bytree':'0.3',
                  'subsample':'0.75'
                 }
    
    # classifier
    xgbc = xgb.XGBClassifier(objective='binary:logistic', parameters=parameters, num_boost_round=10)
    
    # train model
    model = xgbc.fit(X_train, y_train)
    
    
    print('training xgboost classifier ended')
    print('time took {} seconds'.format(time.time() - start_time))
    
    return model

# return a dictionary of result
def predict(model, df):
    # make prediction on test
    test = df[df['eval_set'] == 'test'].copy()
    test = test.drop(['eval_set', 'order_id', 'reordered'], axis=1)
    
    test_pred = (model.predict_proba(test.drop(['user_id', 'product_id'], axis=1))[:,1] >= 0.21).astype(int)
    
    # save result on submit df
    submit = df[df['eval_set'] == 'test'].copy()
    submit['predict'] = test_pred
    
    # a dictionary to save submission
    submit_map = {}
    
    # first give every order id a empty basket
    for od_id in set(submit['order_id']):
        submit_map[od_id] = []
    
    # now we get the dataset of reordered
    reordered_test_df = submit[submit['reordered'] == 1]
    
    # now collecting result for submission
    print('now collecting result for submission')
    for row in tqdm(reordered_test_df.iloc):
        submit_map[row.order_id].append(row.product_id)
        
    return submit_map
        
# save dictionary to submission.csv
def save_result(submit_map):
    f = open('submission.csv', 'w')

    f.write('order_id,products\n')

    for i in tqdm(submit_map.items()):
        f.write('{},'.format(i[0]))

        # if order basket is empty
        if len(i[1]) == 0:
            f.write('None\n')

        else:
            FIRST = True
            for p in i[1]:
                if FIRST:
                    FIRST = False
                    f.write('{}'.format(p))
                else:
                    f.write(' {}'.format(p))
            f.write('\n')

    f.close()

if __name__ == '__main__':
    # check if input data exist
    _csv = glob.glob('data/*.csv')
    if len(_csv) == 0:
        print('Input data not exist, please put \'uxp.csv\' into data/')
        exit(0)
    
    # read training data
    df = pd.read_csv('data/uxp.csv')
    print('input data loaded')
    
    # get number of user and product
#     N_USER = np.max(df['user_id'])
#     N_PRODUCT = np.max(df['product_id'])
    
    # run model
    df = norm_feature(df)
    model = train_xgb(df)
    # make submission.csv
    results = predict(model, df)
    save_result(results)