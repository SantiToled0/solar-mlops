"""Data loading and basic cleaning."""

import duckdb
import pandas as pd


def load_pvdaq_data(
    start_year: int = 2019,
    end_year: int = 2023,
    system_id: int = 10,
) -> pd.DataFrame:
    """Load PVDAQ data from the OEDI S3 data lake."""
    pvdata_urls = [
        (
            f"s3://oedi-data-lake/pvdaq/parquet/pvdata/"
            f"system_id={system_id}/year={year}/month=*/day=*/*.parquet"
        )
        for year in range(start_year, end_year + 1)
    ]

    conn = duckdb.connect()
    conn.execute("LOAD httpfs;")

    result = conn.sql(f"""
        SELECT
            utc_measured_on AS timestamp,

            MAX(value) FILTER (WHERE metric_id = 421) AS irradiance,
            MAX(value) FILTER (WHERE metric_id = 422) AS dc_power,
            MAX(value) FILTER (WHERE metric_id = 423) AS ac_power,
            MAX(value) FILTER (WHERE metric_id = 428) AS ambient_temp,
            MAX(value) FILTER (WHERE metric_id = 429) AS module_temp_1,
            MAX(value) FILTER (WHERE metric_id = 430) AS module_temp_2,
            MAX(value) FILTER (WHERE metric_id = 431) AS module_temp_3

        FROM read_parquet({pvdata_urls})

        GROUP BY utc_measured_on
        ORDER BY utc_measured_on;
    """)

    return result.df()


def clean_pvdaq_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning to PVDAQ measurements."""
    result = df.copy()

    result["timestamp"] = pd.to_datetime(result["timestamp"])

    for column in ["irradiance", "dc_power", "ac_power"]:
        result[column] = result[column].clip(lower=0)

    return result