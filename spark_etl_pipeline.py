"""Spark ETL pipeline for Uber trip data.

Reads raw CSV, applies schema-aware cleansing/derivations, and writes curated
Parquet datasets for downstream analytics and dashboard consumption.
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

REQUIRED_COLS = [
    "START_DATE",
    "END_DATE",
    "CATEGORY",
    "START",
    "STOP",
    "MILES",
    "PURPOSE",
]


def build_spark(app_name: str = "uber-trip-etl") -> SparkSession:
    # On Windows, configure Spark to work without Hadoop binaries
    # by using local filesystem instead of Hadoop
    builder = SparkSession.builder.appName(app_name)
    builder.config("spark.sql.session.timeZone", "UTC")
    builder.config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
    builder.config("spark.sql.sources.partitionOverwriteMode", "dynamic")
    return builder.getOrCreate()


def validate_columns(df) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def transform(df):
    validate_columns(df)

    cleaned = (
        df.select(*REQUIRED_COLS)
        .withColumn("START_DATE", F.to_timestamp(F.col("START_DATE"), "yyyy-MM-dd HH:mm"))
        .withColumn("END_DATE", F.to_timestamp(F.col("END_DATE"), "yyyy-MM-dd HH:mm"))
        .withColumn("MILES", F.col("MILES").cast(DoubleType()))
        .withColumn("PURPOSE", F.coalesce(F.col("PURPOSE"), F.lit("Unknown")))
        .dropna(subset=["START_DATE", "END_DATE", "MILES"])
        .withColumn("Duration(min)", (F.col("END_DATE").cast("long") - F.col("START_DATE").cast("long")) / 60.0)
        .withColumn("Date", F.to_date(F.col("START_DATE")))
        .withColumn("Hour", F.hour(F.col("START_DATE")))
        .withColumn("Weekday", F.date_format(F.col("START_DATE"), "EEEE"))
    )

    daily = (
        cleaned.groupBy("Date")
        .agg(
            F.count(F.lit(1)).alias("trip_count"),
            F.sum("MILES").alias("total_miles"),
            F.avg("Duration(min)").alias("avg_duration_min"),
        )
        .orderBy("Date")
    )

    return cleaned, daily


def write_outputs(cleaned, daily, output_base: str) -> None:
    curated_path = os.path.join(output_base, "curated_trips")
    daily_path = os.path.join(output_base, "daily_aggregates")

    try:
        (
            cleaned.repartition("Date")
            .write.mode("overwrite")
            .partitionBy("Date")
            .parquet(curated_path)
        )
        daily.write.mode("overwrite").parquet(daily_path)
        return
    except Exception as exc:
        # Windows may fail Spark writes when HADOOP_HOME/winutils is unavailable.
        # Keep PySpark transforms but persist parquet through PyArrow fallback.
        message = str(exc)
        if "HADOOP_HOME" not in message and "winutils" not in message:
            raise

    os.makedirs(output_base, exist_ok=True)

    cleaned_pd: pd.DataFrame = cleaned.toPandas()
    daily_pd: pd.DataFrame = daily.toPandas()

    # Use PyArrow's write_to_dataset for consistent schema across partitions
    cleaned_table = pa.Table.from_pandas(cleaned_pd)
    pq.write_to_dataset(
        cleaned_table,
        root_path=curated_path,
        partition_cols=['Date'],
        existing_data_behavior='overwrite_or_ignore'
    )

    daily_table = pa.Table.from_pandas(daily_pd)
    pq.write_to_dataset(
        daily_table,
        root_path=daily_path
    )


def run(input_csv: str, output_base: str) -> None:
    spark = build_spark()
    try:
        raw = spark.read.option("header", True).csv(input_csv)
        cleaned, daily = transform(raw)
        write_outputs(cleaned, daily, output_base)

        # Basic runtime visibility for orchestration logs.
        print(f"Input rows: {raw.count()}")
        print(f"Curated rows: {cleaned.count()}")
        print(f"Daily aggregate rows: {daily.count()}")
        print(f"Wrote curated Parquet to: {os.path.join(output_base, 'curated_trips')}")
        print(f"Wrote daily Parquet to: {os.path.join(output_base, 'daily_aggregates')}")
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Spark ETL for Uber trip data.")
    parser.add_argument("--input", default="Data.csv", help="Path to input CSV file")
    parser.add_argument("--output", default="data", help="Base output directory for parquet datasets")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
