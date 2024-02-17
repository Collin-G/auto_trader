from consts import *
import pandas as pd
import numpy as np
import yfinance as yf
import yahooquery as yq
from fredapi import Fred
import datetime as dt
from scipy import optimize
import logging


from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockQuotesRequest, StockBarsRequest

class MonteCarlo():
    def __init__(self, api, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.api = api
        self.stocks = STOCK_LIST
        self.buys = self.get_weights(7,5000)
        # print(self.buys)
    

    def get_data(self, stocks, start_date, end_date):
    
        # end_date = dt.datetime.now()
        # start_date = end_date - dt.timedelta(days=400)
        data_client = StockHistoricalDataClient(KEY, SECRET)

        request_params = StockBarsRequest(
            symbol_or_symbols=stocks,
            timeframe=TimeFrame.Day,
            adjustment="split",
            start=start_date,
            end=end_date
            )

        bars_df = data_client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)
        stock_data = pd.DataFrame(bars_df)
        stock_data = stock_data.groupby(['symbol', 'timestamp']).mean().unstack(level=0)
        open_prices = stock_data["open"]
        # print(np.dot(open_prices.iloc[0].tolist(), np.ones(len(open_prices.iloc[0].tolist()))))
        log_returns = np.log(open_prices.pct_change()+1)
        # print(log_returns)
        mean_of_returns = log_returns.mean()
        cov_of_returns = log_returns.cov()
        return log_returns , mean_of_returns, cov_of_returns
    
    def get_single_prices(self):
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=400)
        data_client = StockHistoricalDataClient(KEY, SECRET)

        request_params = StockBarsRequest(
            symbol_or_symbols=STOCK_LIST,
            timeframe=TimeFrame.Day,
            start=start_date
            )

        bars_df = data_client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)
        stock_data = pd.DataFrame(bars_df)
        stock_data = stock_data.groupby(['symbol', 'timestamp']).mean().unstack(level=0)
        open_prices = stock_data["open"].iloc[-1]        
        return open_prices

    def portfolio_std(self,weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)
    
    def expected_mc_returns_reparametrized(self,weights, mean_returns, cov_mat, timeframe=30, sim_count=10000, Zs=None):
        mean_mat = np.full(shape =(timeframe, len(weights)), fill_value=mean_returns)
        mean_mat = mean_mat.T

        portfolio_sims = np.full(shape=(timeframe, sim_count),fill_value=0.0)
        initial_value = (float(self.api.get_account().cash))*0.9

        for sim in range(0,sim_count):
            #Z = np.random.normal(size=(timeframe,len(weights))) # np.random.normal(size=(sim_count, timeframe,len(weights)))
            Z = Zs[sim]
            L = np.linalg.cholesky(cov_mat)
            daily_returns = mean_mat + np.inner(L,Z)

            portfolio_sims[:,sim] = np.cumprod(np.inner(weights, (np.exp(daily_returns)-1).T)+1)*initial_value

        expected = portfolio_sims[timeframe-1,:].mean()/initial_value
        return expected
    
    def get_tickers(self,stock_names):
        symbols = " ".join(stock_names)
        tickers = yq.Ticker(symbols)
        return tickers.key_stats
    
    def get_key_stats(self, tickers):
        pegs = []
        insiders = []
        profits = []
        # print(tickers.key_stats)
        for s in tickers:
            print(tickers)
            peg = tickers[s]["pegRatio"]
            pegs.append(1/peg)

            insider = tickers[s]["heldPercentInsiders"]
            insiders.append(insider)

            profit = tickers[s]["profitMargins"]
            profits.append(profit)

        return np.array(pegs), np.array(insiders), np.array(profits) 
    
    def get_sharpe_reparametrized(self,weights, log_returns,cov_mat, risk_free_rate, timeframe=30, sim_count=10000, Zs=None):
        portfolio_value = np.dot(log_returns.mean(), weights)
        return (self.expected_mc_returns_reparametrized(weights, portfolio_value, cov_mat, timeframe=timeframe, sim_count=sim_count, Zs=Zs)- (1+risk_free_rate/(365/timeframe)))/self.portfolio_std(weights,cov_mat)
    

    def get_confidence(self, weights, log_returns, cov_mat, risk_free_rate, insiders, peg,profit, timeframe=30, sim_count=10000, Zs=None):
        sharpe = self.get_sharpe_reparametrized(weights, log_returns,cov_mat, risk_free_rate, timeframe, sim_count, Zs)
        # peg = np.dot(peg, weights)
        # insiders = np.dot(insiders, weights)
        # profit = np.dot(profit, weights)
        return -sharpe
        # return -(sharpe*peg*insiders*profit)

    def get_weights(self, timeframe, sim_count):
        # end_date = dt.datetime.now()
        # start_date = end_date - dt.timedelta(days=100)
        end_date = self.end_date
        start_date = self.start_date

        log_returns, mean_returns, cov_returns = self.get_data(STOCK_LIST, start_date, end_date)

        weights = np.ones(len(mean_returns))
        weights = weights/np.sum(weights)


        constraints = {"type": "eq", "fun": lambda weights : np.sum(weights)- 1}
        bounds = [(0,1) for stock in range(len(STOCK_LIST))]

        Zs = np.random.normal(size=(sim_count, timeframe, weights.size))
       

        insiders = []
        peg = []
        profit = []

        # tickers = self.get_tickers(STOCK_LIST)
        # peg, insiders, profit = self.get_key_stats(tickers)
        optimized_weights_reparamed = optimize.minimize(self.get_confidence, weights, args=(log_returns,cov_returns, RRF, insiders, peg, profit,timeframe, sim_count, Zs), method = "SLSQP", constraints=constraints, bounds=bounds)
        weights  = np.array(optimized_weights_reparamed["x"])
       
        rounded =  np.around(weights, 2)
        weights = rounded
        single_prices = self.get_single_prices()
        dataset = pd.DataFrame({'weights': rounded, "prices": single_prices}, columns=['weights', "prices"])
        mask = dataset["weights"] == 0
        dataset = dataset[~mask]
        print(dataset)
        return dataset


# weeks = 200
# initial_date = dt.datetime.now()- dt.timedelta(weeks=weeks)
# # MC = MonteCarlo(API, initial_date, dt.datetime.now())
# date = initial_date
# total = 0
# for i in range(weeks):
#     MC = MonteCarlo(API, date, date + dt.timedelta(weeks=1))
#     data_client = StockHistoricalDataClient(KEY, SECRET)

#     request_params = StockBarsRequest(
#         symbol_or_symbols=MC.buys.index.values.tolist(),
#         timeframe=TimeFrame.Day,
#         start=date,
#         end = date + dt.timedelta(weeks=2)

#         )
    
#     bars_df = data_client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)
#     stock_data = pd.DataFrame(bars_df)
#     stock_data = stock_data.groupby(['symbol', 'timestamp']).mean().unstack(level=0)
#     open_prices = stock_data["open"]

#     first = np.dot(open_prices.iloc[0].tolist(), MC.buys["weights"].tolist())
#     last = np.dot(open_prices.iloc[-1].tolist(), MC.buys["weights"].tolist())

#     gain = last/first
#     total += gain
#     date += dt.timedelta(weeks=1)
#     print(total/i)
    
