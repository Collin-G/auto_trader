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
        self.headlines = self.get_headlines()
        self.model = self.build_model(num_past_prices,embeddings_dim)
        self.train_model()
        self.predi
    
    def build_model(self, num_past_prices, embeddings_dim):
        # Constants
        max_len = 30
        max_tokens = 1000

        # Gather tf dataset to adapt TextVectorizer
        sample_historical_headlines = ["Hoo Hoo", "HOO", "WHOSE TOES", "HOO", "Hoo hoo hoo"] # replace with array of historical data
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
        input_gains = Input(shape=(num_past_prices), name="input_gains")
        inp_reshaped = Reshape((num_past_prices, 1))(input_gains)
        #g_lstm_1 = ConvLSTM1D(filters=32, kernel_size=5, padding='same')(inp_reshaped) TODO
        g_lstm_1 = LSTM(units=64, return_sequences=True)(inp_reshaped)
        g_lstm_2 = LSTM(units=32, return_sequences=False)(g_lstm_1)

        # Dense
        g_dense_1 = Dense(units=64, activation='relu')(g_lstm_2)
        g_dense_2 = Dense(units=30)(g_dense_1)

        # Concat with headlines embeddings
        concatted_gains_embeds = concatenate([embedded_headlines, g_dense_2])

        # output dense layers
        out_dense_1 = Dense(units = 1, name="gains")(concatted_gains_embeds)
        out_dense_2 = Dense(units = 1, name="vars")(concatted_gains_embeds)

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
        lookback = 5
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=200)
        data = self.get_data(start_date,end_date)
        avg_data = np.convolve(data.values[:,0], np.ones(lookback), mode="valid")/lookback

        for i in range(interval, len(avg_data)):
            gain_in.append(avg_data[i+1-interval:i-1])
            gain_out.append(avg_data[i])
            
            news_date = data.index[i].split(" ")[0]
            headline = self.get_headlines(news_date,news_date)
            news_in.append(headline)
            vol_out.append(data.var().values)


        return gain_in, news_in, gain_out, vol_out
    
    def train_model(self):
        gain_in, news_in, gain_out, vol_out = self.make_training_data()

        # Train
        news_in = np.array(news_in, dtype=str) 
        self.model.fit(
            {"input_headlines": news_in, "input_gains": gain_in},
            {"gains": gain_out, "vars": vol_out},
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
        # print(log_returns)
        mean_of_returns = log_returns.mean()
        cov_of_returns = log_returns.cov()
        return log_returns , mean_of_returns, cov_of_returns
    
    def get_headlines(self, start_date,end_date):
        finnhub_client = finnhub.Client(api_key=FINN_KEY)
        # end_date = dt.datetime.now()
        # start_date = end_date - dt.timedelta(days=20)
        headlines = []
        for s in STOCK_LIST: 
            headlines.append(finnhub_client.company_news(s, _from=start_date.strftime("%Y-%m-%d"), to=end_date.strftime("%Y-%m-%d"))[0].get("headline"))
        return headlines
    
    def predictions(self):
        headlines = self.get_headlines(dt.datetime.now(), dt.datetime.now())
        gains = self.get_data(STOCK_LIST, dt.datetime.now(), dt.datetime.now())
        prediction_batch = self.model.predict([headlines, gains])


