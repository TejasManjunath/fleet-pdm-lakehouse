# Databricks notebook source
df_train = spark.table("fleet_pdm.silver.aps_train")
df_test = spark.table("fleet_pdm.silver.aps_test")

print(f"Train: {df_train.count():,} rows, {len(df_train.columns)} columns")
print(f"Test: {df_test.count():,} rows, {len(df_test.columns)} columns")

# "class" (pos/neg) is now redundant with "label" (1/0) — drop it from the model-ready table
# We keep it in Silver for traceability, but Gold should be strictly numeric and model-ready
df_train_gold = df_train.drop("class")
df_test_gold = df_test.drop("class")

print(f"Train: {len(df_train_gold.columns)} columns (was {len(df_train.columns)})")

from pyspark.sql.types import DoubleType, IntegerType

non_numeric = [f.name for f in df_train_gold.schema.fields 
               if not isinstance(f.dataType, (DoubleType, IntegerType))]

print(f"Non-numeric columns remaining: {non_numeric}")

# Everything except "label" is a feature
feature_cols = [c for c in df_train_gold.columns if c != "label"]
print(f"Total feature columns: {len(feature_cols)}")

(
    df_train_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("fleet_pdm.gold.aps_train")
)
(
    df_test_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("fleet_pdm.gold.aps_test")
)

print("✓ Written: fleet_pdm.gold.aps_train")
print("✓ Written: fleet_pdm.gold.aps_test")

spark.sql("SELECT label, COUNT(*) as n FROM fleet_pdm.gold.aps_train GROUP BY label").display()
spark.sql("SELECT label, COUNT(*) as n FROM fleet_pdm.gold.aps_test GROUP BY label").display()