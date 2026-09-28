import duckdb
import pandas as pd
import matplotlib.pyplot as plt

pvdata_urls = [
    (
        f"s3://oedi-data-lake/pvdaq/parquet/pvdata/"
        f"system_id=10/year={year}/month=*/day=*/*.parquet"
    )
    for year in range(2019, 2024)
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

df = result.df()

print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())

# check temporal gaps
print(df["timestamp"].diff().value_counts().head(10))

# locate temporal gaps
gaps = df.loc[
    df["timestamp"].diff() > pd.Timedelta(minutes=1),
    "timestamp"
]
print(gaps)


# plt.figure()
# plt.plot(df["timestamp"], df["irradiance"])
# plt.xlabel("Time")
# plt.ylabel("Irradiance (W/m²)")
# plt.title("Solar Irradiance")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("irradiance.png")


# plt.figure()
# plt.plot(df["timestamp"], df["ac_power"])
# plt.xlabel("Time")
# plt.ylabel("AC Power (W)")
# plt.title("AC Power")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("ac_power.png")


# check for negative values in irradiance, dc power and ac power
# print("Irradiancia negativa:")
# print((df["irradiance"] < 0).sum())
# print(df.loc[df["irradiance"] < 0, "irradiance"].describe())

# print("DC power negativos:")
# print((df["dc_power"] < 0).sum())
# print(df.loc[df["dc_power"] < 0, "dc_power"].describe())

# print("AC power negativos:")
# print((df["ac_power"] < 0).sum())
# print(df.loc[df["ac_power"] < 0, "ac_power"].describe())


# fixing negative values for irradiance, dc and ac power
df["irradiance"] = df["irradiance"].clip(lower=0)
df["dc_power"] = df["dc_power"].clip(lower=0)
df["ac_power"] = df["ac_power"].clip(lower=0)

# check if every negative value has changed to 0
print(df[["irradiance", "dc_power", "ac_power"]].min())


# mean of the three module temperatures
df["module_temp_mean"] = df[
    ["module_temp_1", "module_temp_2", "module_temp_3"]
].mean(axis=1)

# remove the original module temperature columns
df = df.drop(columns=[
    "module_temp_1",
    "module_temp_2",
    "module_temp_3"
])

df.to_parquet(
    "data/pvdaq_system10_2019_2023.parquet",
    index=False
)
