from datetime import datetime
from datetime import timedelta

from afi.experiments.experiment_configs.erik_nn import NNTrades
from afi.experiments.sliding_configs import TEST_CONFIG
from afi.io.data_readers import ParametricDataReader
from afi.utils.preprocessing_utils import DataSource
from fetcher.preprocessing.tardis.trades import TradesAggregator

sliding_config = config = TEST_CONFIG
ExperimentConfig = NNTrades

start = datetime(2021, 8, 24)
end = datetime(2021, 8, 27)
data_source = DataSource("trades", "BTC_USDT", clock_type=None, downsampling=None)
raw_trades_df = (
    ParametricDataReader.Config(
        data_type=data_source.data_type,
        start=start,
        end=end,
        pair=data_source.pair,
        market_type=data_source.market_type,
        clock_type=data_source.clock_type,
        downsample=data_source.downsampling,
        exchange_source=data_source.exchange,
        file_extension="parquet",
    )
        .instantiate_target()
        .read()
)

data_source = DataSource("agg_trades", "BTC_USDT", clock_type='time_clock_60s', downsampling='1m')
agg_trades_df = (
    ParametricDataReader.Config(
        data_type=data_source.data_type,
        start=start,
        end=end,
        pair=data_source.pair,
        market_type=data_source.market_type,
        clock_type=data_source.clock_type,
        downsample=data_source.downsampling,
        exchange_source=data_source.exchange,
        file_extension="parquet",
    )
        .instantiate_target()
        .read()
)

data_source = DataSource("book_snapshot_25", "BTC_USDT", clock_type=None, downsampling='1m')
lob_snapshot_25_df = (
    ParametricDataReader.Config(
        data_type=data_source.data_type,
        start=start,
        end=end,
        pair=data_source.pair,
        market_type=data_source.market_type,
        clock_type=data_source.clock_type,
        downsample=data_source.downsampling,
        exchange_source=data_source.exchange,
        file_extension="parquet",
    )
        .instantiate_target()
        .read()
)

trades_aggregator = TradesAggregator(agg_clock=timedelta(minutes=1))
len(trades_aggregator.transform(raw_trades_df, lob_snapshot_25_df))
