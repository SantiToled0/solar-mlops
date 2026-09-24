import duckdb
import pandas as pd
import matplotlib.pyplot as plt

pvdata_url = (
    "s3://oedi-data-lake/pvdaq/parquet/pvdata/"
    "system_id=10/year=2021/month=*/day=*/*.parquet"
)

conn = duckdb.connect()

#conn.execute("INSTALL httpfs;")
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

    FROM read_parquet('{pvdata_url}')

    GROUP BY utc_measured_on
    ORDER BY utc_measured_on;
""")

df = result.df()

print(df.head())
print(df.info())
print(df["timestamp"].diff().value_counts().head(10))
gaps = df.loc[df["timestamp"].diff() > pd.Timedelta(minutes=1), "timestamp"]
print(gaps)

print(df.describe())
print(df.isna().sum())

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

#check for negatives values in irradiance, dc power and ac power
# print('Irradiancia negativa:')
# print((df["irradiance"] < 0).sum())
# print(df.loc[df["irradiance"] < 0, "irradiance"].describe())

# print("DC power negativos:")
# print((df["dc_power"] < 0).sum())
# print(df.loc[df["dc_power"] < 0, "dc_power"].describe())

# print("AC power negativos:")
# print((df["ac_power"] < 0).sum())
# print(df.loc[df["ac_power"] < 0, "ac_power"].describe())

#fixing negative values for irradiance, dc and ac power, from <0 to =0
df["irradiance"] = df["irradiance"].clip(lower=0)
df["dc_power"] = df["dc_power"].clip(lower=0)
df["ac_power"] = df["ac_power"].clip(lower=0)

#check if every negative value has changed to 0
print(df[["irradiance", "dc_power", "ac_power"]].min())

# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# fig, ax1 = plt.subplots(figsize=(12, 6))

# # --- Eje izquierdo: potencia ---
# ax1.fill_between(
#     df["timestamp"],
#     0,
#     df["dc_power"],
#     color="blue",
#     alpha=0.20,
#     label="DC Power",
#     zorder=2
# )

# ax1.fill_between(
#     df["timestamp"],
#     0,
#     df["ac_power"],
#     color="green",
#     alpha=0.30,
#     label="AC Power",
#     zorder=3
# )

# ax1.plot(
#     df["timestamp"],
#     df["dc_power"],
#     color="blue",
#     zorder=4
# )

# ax1.plot(
#     df["timestamp"],
#     df["ac_power"],
#     color="green",
#     zorder=5
# )

# ax1.set_xlabel("Time")
# ax1.set_ylabel("Power (W)")

# # --- Eje derecho: irradiancia ---
# ax2 = ax1.twinx()

# ax2.fill_between(
#     df["timestamp"],
#     0,
#     df["irradiance"],
#     color="orange",
#     alpha=0.20,
#     label="Irradiance",
#     zorder=1
# )

# ax2.plot(
#     df["timestamp"],
#     df["irradiance"],
#     color="orange",
#     zorder=2
# )

# ax2.set_ylabel("Irradiance (W/m²)")

# # --- Eje X ---
# locator = mdates.AutoDateLocator(minticks=8, maxticks=10)
# formatter = mdates.ConciseDateFormatter(locator)

# ax1.xaxis.set_major_locator(locator)
# ax1.xaxis.set_major_formatter(formatter)

# # --- Grid ---
# ax1.grid(True, alpha=0.3)

# # --- Leyenda ---
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()

# ax1.legend(
#     lines1 + lines2,
#     labels1 + labels2,
#     loc="upper left"
# )

# plt.title("Solar Irradiance and Power")
# plt.tight_layout()
# plt.savefig("solar_data.png", dpi=150)