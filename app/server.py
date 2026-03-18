import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["PATH"] += os.pathsep + "C:/hadoop/bin"

from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


# ── Make sure we can import from project root ──────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(BASE, "model")
LABELS_PATH = os.path.join(MODEL_PATH, "labels.txt")
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
STATIC_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# ── Spark + model (loaded once at startup) ────────────────────────────────────
print("\n🔥  Starting Spark …")
spark = SparkSession.builder \
    .appName("MusicClassifier-Server") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print(f"📦  Loading model from: {MODEL_PATH}")
model = PipelineModel.load(MODEL_PATH)

# Load label mapping (index → genre name)
with open(LABELS_PATH) as f:
    labels = [line.strip() for line in f.readlines()]

print(f"🏷️   Labels loaded: {labels}")
print("🌐  Server ready!\n")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json(force=True)
    lyrics = data.get("lyrics", "").strip()

    if not lyrics:
        return jsonify({"error": "No lyrics provided"}), 400

    # Build a one-row dataframe
    df     = spark.createDataFrame([(lyrics,)], ["lyrics"])
    result = model.transform(df)

    # probability vector is ordered by StringIndexer label indices
    prob_vector = result.select("probability").first()[0].toArray().tolist()

    # Map index → genre name
    genre_probs = {labels[i]: round(prob_vector[i] * 100, 2)
                   for i in range(len(labels))}

    # Top prediction
    top_genre = max(genre_probs, key=genre_probs.get)

    return jsonify({
        "top_genre"   : top_genre,
        "probabilities": genre_probs
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)