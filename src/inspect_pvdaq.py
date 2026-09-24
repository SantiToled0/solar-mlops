import duckdb

pvdata_url = 'https://oedi-data-lake.s3.amazonaws.com/pvdaq/parquet/pvdata/system_id=10/year=2023/month=1/day=1/system_10__date_2023_01_01.snappy.000.parquet'

metrics_url = 'https://oedi-data-lake.s3.amazonaws.com/pvdaq/parquet/metrics/metrics__system_10__part000.parquet'

conn = duckdb.connect()

conn.execute('INSTALL httpfs;')
conn.execute('LOAD httpfs;')

result = conn.sql(f'''
    SELECT
        m.metric_id,
        m.sensor_name,
        m.common_name,
        m.units
    FROM read_parquet('{metrics_url}') AS m
    WHERE m.metric_id BETWEEN 421 AND 434
    ORDER BY m.metric_id;
''')

print(result)