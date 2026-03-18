# check_model.py
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import os

os.environ["HADOOP_HOME"] = "C:\\hadoop"

spark = SparkSession.builder.appName("check").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

model = PipelineModel.load("model/music_model")
print("Stages:", model.stages)
print("Labels:", model.stages[0].labels)

# Test with lyrics only
df = spark.createDataFrame([("love you baby heart soul forever",)], ["lyrics"])
result = model.transform(df)
result.select("prediction", "probability").show()
spark.stop()