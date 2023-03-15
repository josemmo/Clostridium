import subprocess
from flask import Flask, render_template, request
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/browse_file", methods=["GET", "POST"])
def browse_file():
    if request.method == "POST":
        file_path = request.form["file_path"]
        dataset_label = file_path
        dataset_label.replace("/", "\\")
        return render_template("index.html", dataset_label=dataset_label)
    return render_template("index.html")


@app.route("/browse_directory", methods=["GET", "POST"])
def browse_directory():
    if request.method == "POST":
        storing_path = request.form["storing_path"]
        storing_path.replace("/", "\\")
        results_label = storing_path
        return render_template("index.html", results_label=results_label)
    return render_template("index.html")


@app.route("/preprocessing_apply", methods=["GET", "POST"])
def preprocessing_apply():
    if request.method == "POST":
        file_path = request.form["file_path"]
        storing_path = request.form["storing_path"]
        if "preprocessing_var" in request.form:
            subprocess.call(
                [
                    "Rscript",
                    "--vanilla",
                    "preprocess_maldi.R",
                    file_path,
                    storing_path,
                    "0",
                ]
            )
        return render_template("index.html")
    return render_template("index.html")


@app.route("/model1_apply", methods=["GET", "POST"])
def model1_apply():
    if request.method == "POST":
        storing_path = request.form["storing_path"]
        if "model1_var" in request.form:
            subprocess.call(
                [
                    "conda run -n clostridium",
                    ".\\predictRT.py",
                    "--maldi_path",
                    storing_path,
                ]
            )
        return render_template("index.html")
    return render_template("index.html")


@app.route("/model2_apply", methods=["GET", "POST"])
def model2_apply():
    # Add code to run model 2 here
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
