from pykrx import stock
from pykrx import bond
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score

def get_df(start_date, close_date, ticker_num):
    return stock.get_market_ohlcv(start_date, close_date, ticker_num)

def normalize(stock_df):
    sc = MinMaxScaler()
    set_scaled = pd.DataFrame([])
    set_scaled[['고가','저가']] = sc.fit_transform(stock_df[['고가','저가']].values.reshape(-1,1)).reshape(stock_df[['고가','저가']].shape)
    set_scaled[['거래량']] = sc.fit_transform(stock_df[['거래량']])
    set_scaled[['종가']] = sc.fit_transform(stock_df[['종가']])
    set_scaled.set_index(keys=stock_df.index)

    return set_scaled

def train_test_split(df):
    train, test = train_test_split(normalize(df), test_size = 0.2)
    x_train = train[['고가','저가','거래량']].values
    x_test = test[['고가','저가','거래량']].values

    y_train = train['종가'].values
    y_test = test['종가'].values

def train(x_train, y_train):
    model_lnr = LinearRegression()
    model_lnr.fit(x_train, y_train)
    return model_lnr

def predict_test(model, x_test):
    y_pred = model.predict(x_test)
    return y_pred

def evaluation(y_test, y_pred):
    print("MSE",round(mean_squared_error(y_test,y_pred), 3))
    print("RMSE",round(np.sqrt(mean_squared_error(y_test,y_pred)), 3))
    print("MAE",round(mean_absolute_error(y_test,y_pred), 3))
    print("MAPE",round(mean_absolute_percentage_error(y_test,y_pred), 3))
    print("R2 Score : ", round(r2_score(y_test,y_pred), 3))