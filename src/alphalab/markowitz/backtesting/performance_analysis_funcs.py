import pickle

import pandas as pd
from zipline.api import (
    calendars,
    date_rules,
    get_datetime,
    get_open_orders,
    order_target_percent,
    record,
    schedule_function,
    set_commission,
    set_slippage,
    symbol,
    time_rules,
)
from zipline.finance import commission, slippage


# with open('optimal_weights_dict.pickle', 'rb') as handle:
#     optimal_weights_dict = pickle.load(handle)
def analyze(context, perf):
    # Simple plot of the portfolio value
    perf.portfolio_value.plot()


def rebalance(context, data):
    # Get the current trading date
    current_date = data.current_dt.date()

    # #     pdb.set_trace()
    optimal_weights = optimal_weights_dict[pd.Timestamp(current_date)]

    # Execute the trades based on optimal weights
    for asset, weight in optimal_weights[0].items():
        if data.can_trade(asset) and not get_open_orders(asset):
            order_target_percent(asset, weight)


def exec_trades(data, assets, target_percent):
    # Loop through every asset...
    for asset in assets:
        # ...if the asset is tradeable and there are no open orders...
        if data.can_trade(asset) and not get_open_orders(asset):
            # ...execute the order against the target percent
            order_target_percent(asset, target_percent)


def initialize(context):
    schedule_function(
        rebalance,
        date_rules.week_start(),
        time_rules.market_open(),
        calendar=calendars.US_EQUITIES,
    )
    # Set up the commission model to charge us per share and a volume slippage model
    set_commission(us_equities=commission.PerShare(cost=0.0015, min_trade_cost=0.01))
    set_slippage(
        us_equities=slippage.VolumeShareSlippage(volume_limit=0.0025, price_impact=0.01)
    )


def before_trading_start(context, data):
    pass
