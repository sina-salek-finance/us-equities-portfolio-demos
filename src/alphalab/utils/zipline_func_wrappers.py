import zipline
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.data import USEquityPricing

import pandas as pd


class PricingLoader(object):
    def __init__(self, bundle_data):
        self.loader = USEquityPricingLoader(
            bundle_data.equity_daily_bar_reader,
            bundle_data.adjustment_reader,
            zipline.data.fx.FXRateReader,
        )

    def get_loader(self, column):
        if column not in USEquityPricing.columns:
            raise Exception("Column not in USEquityPricing")
        return self.loader


def build_pipeline_engine(bundle_data, trading_calendar):
    pricing_loader = PricingLoader(bundle_data)

    engine = SimplePipelineEngine(
        get_loader=pricing_loader.get_loader,
        #         calendar=trading_calendar.all_sessions,
        asset_finder=bundle_data.asset_finder,
    )

    return engine


def get_pricing(
    data_portal, trading_calendar, assets, start_date, end_date, field="close"
):
    end_dt = pd.Timestamp(end_date.strftime("%Y-%m-%d"))
    start_dt = pd.Timestamp(start_date.strftime("%Y-%m-%d"))

    end_loc = trading_calendar.closes.index.get_loc(end_dt)
    start_loc = trading_calendar.closes.index.get_loc(start_dt)

    return data_portal.get_history_window(
        assets=assets,
        end_dt=end_dt,
        bar_count=end_loc - start_loc,
        frequency="1d",
        field=field,
        data_frequency="daily",
    )
