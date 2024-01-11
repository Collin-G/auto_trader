import logging
import time
import azure.functions as func
import pandas_market_calendars as mcal
from datetime import datetime
from bot import Bot
from consts import *
app = func.FunctionApp()

@app.schedule(schedule="* 30 10 * * WED", arg_name="myTimer", run_on_startup=True,
              use_monitor=False) 
def testing(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function executed.')

    #implement logic
    is_open = market_is_open(datetime.now().strftime("%Y-%m-%d"))
    logging.info("got dates")
    while not is_open:
        time.sleep(24*60*60)
    logging.info("waited until open")
    Bot(API).close_all()
    logging.info("sold stuff")
    time.sleep(2*60*60)
    Bot(API).make_buys()
    logging.info("bought stuff")



def market_is_open(date):
    result = mcal.get_calendar("NYSE").schedule(start_date=date, end_date=date)
    return result.empty == False
