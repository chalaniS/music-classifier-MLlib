from flask import Flask, request, render_template, jsonify  # FIXED: removed 'app' from imports
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import os

# -----------------------------
# 0. Environment
# -----------------------------
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"

# -----------------------------
# 1. Create Flask app  ← THIS WAS MISSING
# -----------------------------
app = Flask(__name__)

# -----------------------------
# 2. Start Spark
# -----------------------------
spark = SparkSession.builder \
    .appName("MusicClassifierWebApp") \
    .config("spark.hadoop.io.nativeio.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# -----------------------------
# 3. Load model
# -----------------------------
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "music_model"))
print(f"Loading model from: {MODEL_PATH}")
model = PipelineModel.load(MODEL_PATH)

# Get genre labels from StringIndexer (stage 0)
labels = model.stages[0].labels
print(f"Loaded genres: {labels}")

# -----------------------------
# 4. Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    try:
        lyrics = request.form.get("lyrics", "").strip()

        if not lyrics:
            return jsonify({"error": "No lyrics provided"}), 400

        # Create DataFrame with only lyrics column
        df = spark.createDataFrame([(lyrics,)], ["lyrics"])
        result = model.transform(df)
        row = result.select("prediction", "probability").collect()[0]

        prediction_index = int(row["prediction"])
        predicted_genre = labels[prediction_index]

        probs = list(row["probability"])
        prob_dict = {labels[i]: round(probs[i] * 100, 2) for i in range(len(labels))}
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

        return jsonify({
            "predicted_genre": predicted_genre,
            "probabilities": prob_dict
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 5. Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=False, port=5000)