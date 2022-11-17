import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

st.title("STOCK PREDICTION")
user_input = st.text_input("Enter stock ticker", "GAIL.NS")

st.write("Check out this [link](https://finance.yahoo.com/lookup/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAK8-4VlO4FGSQc2HpxyRH38sU0ZoLWISRjzETxDHZ7dKVuTE1jH7GeVh24Zp6fF-52gCi8nMpO_musKo6krHTbD0aN6s1caKSQscEGuRxDHmC0X61I4K7KOiatGEqfTEx0oRlF7tNjuG3znWwjUkg0jlU9itszmDgUgNVs8QDety) for tickers")
data = yf.download(tickers=user_input, period="5y", interval="1d")

st.subheader("Data {}: ".format(user_input))
st.write(data.describe())

opn = data[['Open']]
ds = opn.values


def create_ds(dataset, step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)


normalizer = MinMaxScaler(feature_range=(0, 1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1, 1))
train_size = int(len(ds_scaled)*0.7)
test_size = len(ds_scaled) - train_size
ds_train, ds_test = ds_scaled[0:train_size,
                              :], ds_scaled[train_size: len(ds_scaled), : 1]
time_stamp = 100
X_train, y_train = create_ds(ds_train, time_stamp)
X_test, y_test = create_ds(ds_test, time_stamp)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = load_model('lstm.h5')

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)
fut_inp = ds_test[len(ds_test)-100:]
fut_inp = fut_inp.reshape(1, -1)
tmp_inp = list(fut_inp)
tmp_inp = tmp_inp[0].tolist()
lst_output = []
n_steps = 100
i = 0
while(i < 30):

    if(len(tmp_inp) > 100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp = fut_inp.reshape(1, -1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i = i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i+1

plot_new = np.arange(1, 101)
plot_pred = np.arange(101, 131)

fig1 = plt.figure(figsize=(12, 6))
st.subheader("Prediciton")
plt.plot(plot_new, normalizer.inverse_transform(
    ds_scaled[len(ds_scaled)-100:]))
plt.plot(plot_pred, normalizer.inverse_transform(lst_output))
plt.ylabel("Price")
plt.xlabel("Time")
plt.legend()
st.pyplot(fig1)

ds_new = ds_scaled.tolist()
ds_new.extend(lst_output)
final_graph = normalizer.inverse_transform(ds_new).tolist()
fig2 = plt.figure(figsize=(12, 6))
st.subheader("{0} prediction of next month open".format(user_input))
plt.plot(final_graph,)
plt.ylabel("Price")
plt.xlabel("Time")
plt.axhline(y=final_graph[len(final_graph)-1], color='red', linestyle=':',
            label='NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]), 2)))
plt.legend()
st.pyplot(fig2)
