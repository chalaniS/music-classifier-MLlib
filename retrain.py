from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os, shutil

os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["PATH"] += ";C:/hadoop/bin"

spark = SparkSession.builder \
    .appName("MusicClassifier-Retrain") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

BASE     = os.path.dirname(__file__)
DATA     = os.path.join(BASE, "data", "Merged_dataset.csv")
MODEL    = os.path.join(BASE, "model")

print("\n🔄  RE-TRAINING on 8-class Merged dataset …")
print(f"📂  Data : {DATA}")
print(f"💾  Model: {MODEL}\n")

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("multiLine", "true") \
    .option("escape", '"') \
    .csv(DATA) \
    .select("lyrics", "genre") \
    .na.drop()

print(f"Rows loaded: {df.count()}")
print("Genre distribution:")
df.groupBy("genre").count().orderBy("count", ascending=False).show()

labelIndexer = StringIndexer(inputCol="genre", outputCol="label", handleInvalid="keep")
tokenizer    = Tokenizer(inputCol="lyrics", outputCol="words")
remover      = StopWordsRemover(inputCol="words", outputCol="filtered")
hashTF       = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20000)
idf          = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2)
lr           = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.0)

pipeline = Pipeline(stages=[labelIndexer, tokenizer, remover, hashTF, idf, lr])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train_df)

preds    = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
print(f"\n✅  Test Accuracy: {evaluator.evaluate(preds) * 100:.2f}%")

if os.path.exists(MODEL):
    shutil.rmtree(MODEL)
model.save(MODEL)

labels = model.stages[0].labels
with open(os.path.join(MODEL, "labels.txt"), "w") as f:
    f.write("\n".join(labels))

print(f"🏷️   Labels: {list(labels)}")
spark.stop()
print("\n🎉  Retraining complete!\n")