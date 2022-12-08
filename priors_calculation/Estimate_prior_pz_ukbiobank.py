# This script calculate the beta and se using the estimated and reported pvalues in uk biobank studies. 
# These are used as priors to do QC on other GWAS studies
# Approx 2K UK Biobank studies are considered

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pyspark.sql.types as t
import scipy as sc
from scipy import stats
import numpy as np
from pyspark import SparkContext as spc

#warnings.simplefilter(action='ignore', category=FutureWarning)

spark = SparkSession.builder.master("yarn").getOrCreate()

def calculate_pval(z_score):
       return(float(sc.stats.chi2.sf((z_score), 1)))

calculate_pval_udf = f.udf(calculate_pval, t.DoubleType())

def calculate_lin_reg(y, x):
    lin_reg = sc.stats.linregress(y,x)
    return [float(lin_reg.slope), float(lin_reg.stderr), float(lin_reg.intercept), float(lin_reg.intercept_stderr)]

lin_udf = f.udf(calculate_lin_reg,  linear_reg_Schema)

path_ukbio = "gs://genetics-portal-dev-sumstats/filtered/significant_window_2mb/gwas/NEALE2_*"
NEALE_studies = spark.read.parquet(path_ukbio)

NEALE_studies = (
    NEALE_studies
        .withColumn("zscore", f.col("beta")/f.col("se"))
        .withColumn("new_pval", calculate_pval_udf(f.col("zscore")**2))
        .select("study_id","pval", "new_pval")
)

linear_reg_Schema = t.StructType([
    t.StructField("beta", t.FloatType(), False),
    t.StructField("beta_stderr", t.FloatType(), False),
    t.StructField("intercept", t.FloatType(), False),
    t.StructField("intercept_stderr", t.FloatType(), False)])

final_data_priors = (NEALE_studies
              .groupBy("study_id")
              .agg(f.collect_list("pval").alias("pval_vector"),
                   f.collect_list("new_pval").alias("new_pval_vector")
                   )
              .withColumn("result_lin_reg", lin_udf(f.col("pval_vector"), f.col("new_pval_vector")))
            )



final_data.write.parquet("/home/ba13/prior_ukbio_pz")

