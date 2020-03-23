import numpy as np
import pandas as pd
import statsmodels.api as sm
import csv
from sklearn.linear_model import LinearRegression

def beratbasahModel(data,feature_names):
    X=data[feature_names]
    y1=data.Berat_Basah
    model = LinearRegression().fit(X, y1)
    model=sm.OLS(y1,X).fit()
    X=sm.add_constant(X)
    model=sm.OLS(y1,X).fit()
    lin_model = np.around(model.params, decimals=3)
    return (lin_model)

def jumlahdaunModel(data,feature_names):
    X=data[feature_names]
    y1=data.Jumlah_Daun
    model = LinearRegression().fit(X, y1)
    model=sm.OLS(y1,X).fit()
    X=sm.add_constant(X)
    model=sm.OLS(y1,X).fit()
    lin_model = np.around(model.params, decimals=3)
    return (lin_model)

def tinggitanamanModel(data,feature_names):
    X=data[feature_names]
    y1=data.Tinggi_Tanaman
    model = LinearRegression().fit(X, y1)
    model=sm.OLS(y1,X).fit()
    X=sm.add_constant(X)
    model=sm.OLS(y1,X).fit()
    lin_model = np.around(model.params, decimals=3)
    return (lin_model)


data = pd.read_excel('data_dummy.xlsx',sheet_name='Sheet2')
feature_names=['Kalium','Nitrogen','Postassium','Air']

berat_model = beratbasahModel(data,feature_names)
berat_model = np.array(berat_model)

daun_model = jumlahdaunModel(data,feature_names)
daun_model = np.array(daun_model)

tinggi_model = tinggitanamanModel(data,feature_names)
tinggi_model = np.array(tinggi_model)

row_list = [
    ["Model", "Kalium", "Nitrogen", "Postassium", "Air", "Constant" ],
    ["tinggi_tanaman", tinggi_model[1], tinggi_model[2], tinggi_model[3], tinggi_model[4], tinggi_model[0]],
    ["jumlah_daun", daun_model[1], daun_model[2], daun_model[3], daun_model[4], daun_model[0]],
    ["berat_basah", berat_model[1], berat_model[2], berat_model[3], berat_model[4], berat_model[0]]
]
with open('model.csv', 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC,
                        delimiter=';')
    writer.writerows(row_list)