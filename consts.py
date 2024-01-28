from alpaca.trading.client import TradingClient
from fredapi import Fred

KEY = "PKQ9XJDJLTX686HE3ZL9"
SECRET = "4ES985YvYlOWz0eAtzdhcETJ1asEPBSw3gq9ZXs7"
ENDP = "https://paper-api.alpaca.markets"
API = TradingClient(KEY, SECRET)
STOCK_LIST = ["AAPL", "AMZN", "GOOG", "JPM", "KO"]
FINN_KEY = "cmov2lhr01qjn6789utgcmov2lhr01qjn6789uu0"
FRED = Fred(api_key="f3fea224d98377beff02b72fbe0cb196")
RRF = (FRED.get_series_latest_release("GS10")).iloc[-1]/100

