# Databricks notebook source
import numpy
import pandas
import os

# File paths in the Unity Catalog volume
TRAIN_PATH = "/Volumes/fleet_pdm/bronze/raw/aps_failure_training_set.csv"
TEST_PATH = "/Volumes/fleet_pdm/bronze/raw/aps_failure_test_set.csv"

# Target catalog and schema for Bronze tables
CATALOG = "fleet_pdm"
SCHEMA = "bronze"

print(f"Reading from: {TRAIN_PATH}")
print(f"Writing to: {CATALOG}.{SCHEMA}")

df_train = (
    spark.read
    .option("header", "true")           # first row is column names
    .option("nullValue", "na")          # Scania encoded missing values as literal "na"
    .option("inferSchema", "false")     # keep everything as string for now
    .csv(TRAIN_PATH)
)

print(f"Training rows: {df_train.count():,}")
print(f"Training columns: {len(df_train.columns)}")
df_train.limit(5).display()

train_table = f"{CATALOG}.{SCHEMA}.aps_train"

(
    df_train.write
    .format("delta")                    # medallion architecture requires Delta
    .mode("overwrite")                  # idempotent — rerunning the notebook replaces
    .option("overwriteSchema", "true")  # allow schema replacement on rerun
    .saveAsTable(train_table)
)

print(f"✓ Written: {train_table}")

df_test = (
    spark.read
    .option("header", "true")
    .option("nullValue", "na")
    .option("inferSchema", "false")
    .csv(TEST_PATH)
)

print(f"Test rows: {df_test.count():,}")
print(f"Test columns: {len(df_test.columns)}")

test_table = f"{CATALOG}.{SCHEMA}.aps_test"

(
    df_test.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(test_table)
)

print(f"✓ Written: {test_table}")

# Confirm both tables exist and are queryable
spark.sql(f"SELECT COUNT(*) AS train_count FROM {CATALOG}.{SCHEMA}.aps_train").display()
spark.sql(f"SELECT COUNT(*) AS test_count FROM {CATALOG}.{SCHEMA}.aps_test").display()

# Peek at class balance in training (the reason cost-sensitive optimization matters)
spark.sql(f"""
    SELECT class, COUNT(*) AS n
    FROM {CATALOG}.{SCHEMA}.aps_train
    GROUP BY class
""").display()