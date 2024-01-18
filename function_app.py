import logging
import time
import azure.functions as func
import pandas_market_calendars as mcal
import datetime
from bot import Bot
from consts import *
app = func.FunctionApp()

@app.schedule(schedule="* 30 10,12 * * WED", arg_name="myTimer", run_on_startup=True,
              use_monitor=False) 
def testing(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function executed.')

    utc_now = datetime.datetime.now()

    #implement logic
    is_open = market_is_open(datetime.now().strftime("%Y-%m-%d"))
    logging.info("got dates")
    if not is_open:
        return
    if utc_now.hour == 10:
        Bot(API).close_all()
        logging.info("sold stuff")
    elif utc_now.hour == 12:    
        Bot(API).make_buys()
        logging.info("bought stuff")



def market_is_open(date):
    result = mcal.get_calendar("NYSE").schedule(start_date=date, end_date=date)
    return result.empty == False
