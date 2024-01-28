import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, GlobalAveragePooling1D, Conv2D, ConvLSTM2D, ConvLSTM1D, Input, Flatten, Reshape, TextVectorization, concatenate
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import pad_sequences
#from tensorflow.keras import ops
import keras
import numpy as np
import pandas as pd

tf.config.run_functions_eagerly(True)
import datetime as dt

from consts import *

import finnhub 
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockQuotesRequest, StockBarsRequest

class Model():
    def __init__(self, num_past_prices, embeddings_dim):
        self.num_past_prices = num_past_prices
        self.embeddings_dim = embeddings_dim
        self.model = self.build_model(num_past_prices,embeddings_dim)
        self.train_model()
        self.predictions()
    
    def build_model(self, num_past_prices, embeddings_dim, max_len=30, max_tokens=100):
        # Gather tf dataset to adapt TextVectorizer
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=200)
        articles = self.get_articles(start_date.strftime('%Y-%m-%d'),end_date.strftime('%Y-%m-%d'))
        headlines = [a.get("headline") for a in articles[0:10]]
        sample_historical_headlines = headlines # replace with array of historical data
        dataset = tf.data.Dataset.from_tensor_slices(sample_historical_headlines)

        """Total arch"""
        # Headline Encoder
        input_headlines = Input(shape=(1, ), dtype=tf.string, name="input_headlines")

        tv_layer = TextVectorization(max_tokens=max_tokens, output_mode='int', output_sequence_length=max_len)
        tv_layer.adapt(dataset.batch(64))

        tv_layer_1 = tv_layer(input_headlines)
        embeddings_1 = Embedding(input_dim=max_tokens+1, output_dim=embeddings_dim, input_length=max_len)(tv_layer_1)
        lstm_1 = LSTM(units=30, return_sequences=False)(embeddings_1) # Default activation tanh
        embedded_headlines = lstm_1 # More to compe perhaps

        # Stock Gain Encoder
        input_gains = Input(shape=(num_past_prices  ), name="input_gains")
        inp_reshaped = Reshape((num_past_prices, 1))(input_gains)
        #g_lstm_1 = ConvLSTM1D(filters=32, kernel_size=5, padding='same')(inp_reshaped) TODO
        g_lstm_1 = LSTM(units=64, return_sequences=True)(inp_reshaped)
        g_lstm_2 = LSTM(units=32, return_sequences=False)(g_lstm_1)

        # Dense
        g_dense_1 = Dense(units=64, activation='relu')(g_lstm_2)
        g_dense_2 = Dense(units=30)(g_dense_1)

        # Concat with headlines embeddings
        concatted_gains_embeds = concatenate([embedded_headlines, g_dense_2])
        post_concat_1 = Dense(units=12, activation='tanh')(concatted_gains_embeds)
        post_concat_2 = Dense(units = 1, activation='sigmoid')(post_concat_1)

        # output dense layers
        out_dense_1 = Dense(units = 1, name="gains")(post_concat_1)
        out_dense_2 = Dense(units = 1, name="vars", activation='relu')(post_concat_2)#, activation='relu')(post_concat_2)#(concatted_gains_embeds)

        model = keras.Model(
            inputs = [input_headlines, input_gains], 
            outputs = {"gains":out_dense_1, "vars":out_dense_2}
        )

        model.compile(optimizer="adam", loss="mean_squared_error")
                #     loss={
                #         "gains":,
                #         "vars":
                #     })
        return model

    
    def make_training_data(self):
        gain_in = []
        news_in = []
        vol_out = []
        gain_out = []
        interval = 7
        lookback = self.num_past_prices+1
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=200)
        data = self.get_data(STOCK_LIST,start_date,end_date)
        data = data.dropna()
        # print(data.values)
        # avg_data = np.array(data.values[:,0], np.ones(interval), mode="valid")/interval
        while(len(data)%interval != 0):
            data.drop(data.tail(1).index,inplace=True)
        avg_data = np.array(data).reshape(-1, interval).mean(axis=1)
        articles = self.get_articles(start_date.strftime('%Y-%m-%d'),end_date.strftime('%Y-%m-%d'))
        for i in range(lookback, len(avg_data)):
            gain_in.append(avg_data[i-lookback:i-1].tolist())
            # print(avg_data[i])
            gain_out.append([avg_data[i]])
            
            news_date = str(data.index[i]).split(" ")[0]
        
            for s in STOCK_LIST:
                for a in articles:
                    # print(len_)
                    # if not a.get("datetime"):
                    #     continue
                    # print(a.get("datetime"))
                    ts = int(a.get("datetime"))
                    date = dt.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                    # print(date)
                    if a.get("related") == s:
                        news_in.append(a.get("headline"))
                        break
            vol_out.append(data.iloc[i-lookback:i].var().values)

        return gain_in, news_in, gain_out, vol_out
    
    def train_model(self):
        gain_in, news_in, gain_out, vol_out = self.make_training_data()
        print(news_in)
        print(gain_in)
        print(gain_out)
        print(vol_out)
        # Train
        news_in = np.array(news_in, dtype=str) 
        # print(news_in.size)
        # print(gain_in.size)
        self.model.fit(
            {"input_headlines": news_in, "input_gains": np.array(gain_in)},
            {"gains": np.array(gain_out), "vars": np.array(vol_out)},
            epochs=2,
            batch_size=32,
        )
    
    
    def get_data(self, stocks, start_date, end_date):
        stocks = STOCK_LIST
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=200)
        data_client = StockHistoricalDataClient(KEY, SECRET)

        request_params = StockBarsRequest(
            symbol_or_symbols=stocks,
            timeframe=TimeFrame.Day,
            start=start_date
            )

        bars_df = data_client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)
        stock_data = pd.DataFrame(bars_df)
        stock_data = stock_data.groupby(['symbol', 'timestamp']).mean().unstack(level=0)
        open_prices = stock_data["open"]
        log_returns = np.log(open_prices.pct_change()+1)
        return log_returns
    
    def get_headlines(self, start_date,end_date):
        finnhub_client = finnhub.Client(api_key=FINN_KEY)
        # end_date = dt.datetime.now()
        # start_date = end_date - dt.timedelta(days=20)
        headlines = []
        for s in STOCK_LIST: 
            headlines.append(finnhub_client.company_news(s, _from=start_date, to=end_date)[0].get("headline"))
        return headlines

    def get_articles(self, start_date,end_date):
        start_date = "2023-01-01"
        finnhub_client = finnhub.Client(api_key=FINN_KEY)
        articles = []
        for s in STOCK_LIST: 
            articles.extend(finnhub_client.company_news(s, _from=start_date, to=end_date))
        # print(articles)
        return articles
    
    def predictions(self):
        interval = 7
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=200)
        headlines = self.get_headlines(end_date.strftime('%Y-%m-%d'), dt.datetime.now().strftime('%Y-%m-%d'))
        data = self.get_data(STOCK_LIST, end_date.strftime('%Y-%m-%d'), start_date.strftime('%Y-%m-%d'))
        
        # gains = np.convolve(data.values[:,0], np.ones(interval), mode="valid")/interval
        while(len(data)%interval != 0):
            data.drop(data.tail(1).index,inplace=True)
        gains = np.array(data).reshape(-1, interval).mean(axis=1)
        
        # print(gains[-6:-1])
        # print(headlines)
        # printdata
        prediction_batch = self.model.predict([np.array([headlines], dtype=str), np.array([gains[-6:-1]])])
        print(prediction_batch)
        return prediction_batch


