# Databricks notebook source
import pyspark 

# Read the Bronze Delta tables we created in the previous notebook
df_train = spark.table("fleet_pdm.bronze.aps_train")
df_test = spark.table("fleet_pdm.bronze.aps_test")

print(f"Train: {df_train.count():,} rows, {len(df_train.columns)} columns")
print(f"Test: {df_test.count():,} rows, {len(df_test.columns)} columns")

from pyspark.sql.functions import col

# Every column except "class" should be numeric
sensor_cols = [c for c in df_train.columns if c != "class"]

for c in sensor_cols:
    df_train = df_train.withColumn(c, col(c).cast("double"))
    df_test = df_test.withColumn(c, col(c).cast("double"))

df_train.printSchema()

from pyspark.sql.functions import when

df_train = df_train.withColumn(
    "label",
    when(col("class") == "pos", 1).otherwise(0).cast("integer")
)
df_test = df_test.withColumn(
    "label",
    when(col("class") == "pos", 1).otherwise(0).cast("integer")
)

df_train.groupBy("label").count().display()

from pyspark.sql.functions import count as spark_count, isnan

total_rows = df_train.count()

missing_pct = []
for c in sensor_cols:
    n_missing = df_train.filter(col(c).isNull()).count()
    missing_pct.append((c, n_missing, round(100 * n_missing / total_rows, 2)))

missing_df = spark.createDataFrame(missing_pct, ["column", "n_missing", "pct_missing"])
missing_df.orderBy(col("pct_missing").desc()).display()

# Build a Python dict {column_name: pct_missing} from what we already computed
missing_lookup = {row["column"]: row["pct_missing"] for row in missing_df.collect()}

THRESHOLD = 5.0  # percent

high_missing_cols = [c for c, pct in missing_lookup.items() if pct > THRESHOLD]
low_missing_cols = [c for c, pct in missing_lookup.items() if pct <= THRESHOLD]

print(f"High-missing columns (get indicator flag): {len(high_missing_cols)}")
print(f"Low-missing columns (impute only): {len(low_missing_cols)}")

# approxQuantile computes the median (0.5 quantile) per column, approximate for speed
medians = {}
for c in sensor_cols:
    med = df_train.approxQuantile(c, [0.5], 0.01)[0]
    medians[c] = med if med is not None else 0.0

print(f"Computed medians for {len(medians)} columns")
# Peek at a few
{k: medians[k] for k in list(medians)[:5]}

from pyspark.sql.functions import lit

# Add indicator flags for high-missing columns (do this BEFORE imputing, on both sets)
for c in high_missing_cols:
    df_train = df_train.withColumn(f"{c}_was_missing", col(c).isNull().cast("integer"))
    df_test = df_test.withColumn(f"{c}_was_missing", col(c).isNull().cast("integer"))

# Now impute — fillna using the training-set medians, applied to BOTH train and test
df_train = df_train.fillna(medians)
df_test = df_test.fillna(medians)

print(f"Train columns after Silver: {len(df_train.columns)}")
print(f"Test columns after Silver: {len(df_test.columns)}")
df_train.select(high_missing_cols[0], f"{high_missing_cols[0]}_was_missing").limit(5).display()

from pyspark.sql.functions import sum as spark_sum

# Count total remaining nulls across ALL columns, not just a sample
null_counts = df_train.select(
    [spark_sum(col(c).isNull().cast("int")).alias(c) for c in df_train.columns]
).collect()[0].asDict()

remaining_nulls = {c: n for c, n in null_counts.items() if n > 0}

if remaining_nulls:
    print(f"⚠️ {len(remaining_nulls)} columns still have nulls:")
    print(remaining_nulls)
else:
    print("✓ Zero nulls remaining in df_train")

(
    df_train.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("fleet_pdm.silver.aps_train")
)

(
    df_test.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("fleet_pdm.silver.aps_test")
)

print("✓ Written: fleet_pdm.silver.aps_train")
print("✓ Written: fleet_pdm.silver.aps_test")

##cell_10
histogram_groups = {
    "ag": [f"ag_{i:03d}" for i in range(10)],
    "ay": [f"ay_{i:03d}" for i in range(10)],
    "az": [f"az_{i:03d}" for i in range(10)],
    "ba": [f"ba_{i:03d}" for i in range(10)],
    "cn": [f"cn_{i:03d}" for i in range(10)],
    "cs": [f"cs_{i:03d}" for i in range(10)],
    "ee": [f"ee_{i:03d}" for i in range(10)],
}

for prefix, cols in histogram_groups.items():
    print(f"{prefix}: {cols}")

##cell_11
from pyspark.sql.functions import col, greatest, when, lit
from functools import reduce

def add_histogram_features(df, prefix, cols):
    total_expr = reduce(lambda a, b: a + b, [col(c) for c in cols])
    weighted_sum_expr = reduce(lambda a, b: a + b, [col(c) * i for i, c in enumerate(cols)])
    max_expr = greatest(*[col(c) for c in cols])

    df = df.withColumn(f"{prefix}_total", total_expr)
    df = df.withColumn(
        f"{prefix}_wmean",
        when(col(f"{prefix}_total") > 0, weighted_sum_expr / col(f"{prefix}_total")).otherwise(lit(0.0))
    )
    df = df.withColumn(
        f"{prefix}_concentration",
        when(col(f"{prefix}_total") > 0, max_expr / col(f"{prefix}_total")).otherwise(lit(0.0))
    )
    return df

for prefix, cols in histogram_groups.items():
    df_train = add_histogram_features(df_train, prefix, cols)
    df_test = add_histogram_features(df_test, prefix, cols)

print(f"Train columns after histogram features: {len(df_train.columns)}")
df_train.select("ag_total", "ag_wmean", "ag_concentration").limit(5).display()

##cell_12
from pyspark.sql.functions import sum as spark_sum

new_cols = [f"{p}_{stat}" for p in histogram_groups for stat in ["total", "wmean", "concentration"]]

null_check = df_train.select(
    [spark_sum(col(c).isNull().cast("int")).alias(c) for c in new_cols]
).collect()[0].asDict()

bad = {c: n for c, n in null_check.items() if n > 0}
print("✓ Clean" if not bad else f"⚠️ Nulls found: {bad}")

##cell_13
(
    df_train.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("fleet_pdm.silver.aps_train")
)
(
    df_test.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("fleet_pdm.silver.aps_test")
)
print("✓ Silver updated with histogram features")