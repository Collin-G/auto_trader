from alpaca.trading.client import TradingClient
from fredapi import Fred

KEY = "PKQ9XJDJLTX686HE3ZL9"
SECRET = "4ES985YvYlOWz0eAtzdhcETJ1asEPBSw3gq9ZXs7"
ENDP = "https://paper-api.alpaca.markets"
API = TradingClient(KEY, SECRET)
STOCK_LIST = ["MSFT", "AAPL", "V", "JPM", "UNH", "WMT", "AMGN", "AMZN", "LMT", "CRM", "AMD", "TSLA", "BA", "MCD", "DIS"]
FRED = Fred(api_key="f3fea224d98377beff02b72fbe0cb196")
RRF = (FRED.get_series_latest_release("GS10")).iloc[-1]/100

#  "V", "JPM", "UNH", "WMT", "JNJ", "PG", "HD", "MRK"
#               "CVX", "CRM", "KO", "MCD", "CSCO", "DIS", "AMD", "VZ", "IBM", "CAT", 
#               "AMGN", "NKE", "AXP", "BA", "HON", "GS", "MMM", "TRV", "LMT"]