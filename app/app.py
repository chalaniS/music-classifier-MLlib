from flask import Flask, request, render_template
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import os

def create_chart(labels, probs):
    plt.figure()
    plt.bar(labels, probs)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    chart_path = "static/chart.png"
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

app = Flask(__name__)

spark = SparkSession.builder.appName("WebApp").getOrCreate()

model = spark.read.load("../model/music_model_8genre")

labels = model.stages[0].labels

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    chart = None

    if request.method == "POST":
        lyrics = request.form["lyrics"]

        df = spark.createDataFrame([(lyrics,)], ["lyrics"])
        result = model.transform(df)

        # Prediction
        prediction_index = result.select("prediction").collect()[0][0]
        predicted_genre = labels[int(prediction_index)]

        # Probabilities
        probs = result.select("probability").collect()[0][0]
        prob_list = list(probs)

        # Create chart
        chart = create_chart(labels, prob_list)

        prediction = predicted_genre

    return render_template("index.html", prediction=prediction, chart=chart)

app.run(debug=True)