import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from sklearn import linear_model
from scipy.sparse import coo_matrix
import datetime
from tqdm import tqdm
import pathlib


project_path = pathlib.Path(__file__).parent.absolute()

app = Flask(__name__)

model = pickle.load(open(str(project_path) +'/model/model.pkl', 'rb'))

# load csv
df = pd.read_csv(str(project_path) + '/dataset/df_withC.csv', index_col=[0])
df.reset_index(drop=True, inplace=True)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df = df[['InvoiceDate','Description','SC_Clean','Quantity','CustomerID']]
df.rename(columns={'SC_Clean':'StockCode'}, inplace=True)

# getting all customer IDs
# all_customers = [df2['CustomerID'][i] for i in range(len(df2['CustomerID']))]

# # getting all stock codes
# all_stockcodes = [df2['StockCode'][i] for i in range(len(df2['StockCode']))]

# Train-test split
start_train = df['InvoiceDate'].min()
start_test = start_train + pd.to_timedelta(15, unit='w')
end_test = start_test + pd.to_timedelta(5, unit='w')

df = df.loc[(df['InvoiceDate'] > start_train) & (df['InvoiceDate'] <= end_test)]

# Create train_split flag
df['train_split'] = (df['InvoiceDate'] <= start_test).astype(int)
print("Proportion of train events: {:.2f}".format(df['train_split'].mean()))


# transform
user_cat = df['CustomerID'].astype('category')
item_cat = df['StockCode'].astype("category")


item_user_train = coo_matrix((df['train_split'],
                              (item_cat.cat.codes,
                               user_cat.cat.codes))).tocsr()

# remove zero entries
item_user_train.eliminate_zeros()

# produce transpose of item_user_train
user_item_train = item_user_train.T

# map each item and user category to a unique numeric code
user_map = dict(zip(user_cat, user_cat.cat.codes))
item_map = dict(zip(item_cat, item_cat.cat.codes))


gloss = df[['StockCode','Description']]
gloss.drop_duplicates(subset=['StockCode'],inplace=True)
gloss.reset_index(drop=True,inplace=True)
gloss.set_index('StockCode',inplace=True)

my_dict = gloss.to_dict()
my_dict = my_dict.pop('Description', None)

def get_keys(value, dictionary):
    """Function to get dictionary keys with specifiec value"""
    return list(dictionary.keys())[list(dictionary.values()).index(value)]

def get_values(key, dictionary):
    """Function to get dictionary values from a specific key"""
    return(dictionary[key])


@app.route('/')
def home():
    return render_template('index_steffen.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_id = list(request.form.values())[0] 
    fin_feature = float(int_id)
    prediction = model.recommend(user_map[fin_feature], user_item_train) 
    new_recs = list(map(lambda x: (get_keys(x[0], item_map), x[1]), prediction))
    
    new_recs_df = pd.DataFrame(new_recs, columns=['stock','score'])
    new_recs_df['description'] = new_recs_df['stock'].apply(lambda x: get_values(x, my_dict))
    new_recs_df = new_recs_df[['stock','description','score']]
    
    #final_pred = list(map(lambda x: (get_values(x[0], gloss), x[1]), new_recs))[:5]
    stockids = []
    names = []
    scores = []

    for i in range(len(new_recs_df)):
        stockids.append(new_recs_df.iloc[i][0])
        names.append(new_recs_df.iloc[i][1])
        scores.append(round(new_recs_df.iloc[i][2],2))  

    output = new_recs
    return render_template('positive.html',stock_id_=stockids,names_=names,scores_=scores)
    #return render_template('index.html', prediction_text='Recommended Items: /n{0}.'.format(output))

@app.route('/predict_2',methods=['POST'])
def predict_2():

    int_id = list(request.form.values())[0]
    item_id = float(int_id)
    related = model.similar_items(item_map[item_id])
    item_rec = list(map(lambda x: (get_keys(x[0], item_map), x[1]), related))

    new_recs_df = pd.DataFrame(item_rec, columns=['stock','score'])
    new_recs_df['description'] = new_recs_df['stock'].apply(lambda x: get_values(x, my_dict))
    new_recs_df = new_recs_df[['stock','description','score']]
    
    #final_pred = list(map(lambda x: (get_values(x[0], gloss), x[1]), new_recs))[:5]
    stockids = []
    names = []
    scores = []

    for i in range(len(new_recs_df)):
        stockids.append(new_recs_df.iloc[i][0])
        names.append(new_recs_df.iloc[i][1])
        scores.append(round(new_recs_df.iloc[i][2],2))  


    return render_template('positive.html',stock_id_=stockids,names_=names,scores_=scores)


if __name__ == "__main__":
    app.run(debug=True)