import subprocess
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os

ALLOWED_EXTENSIONS = set(["zip"])


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"


@app.route("/")
def index():
    return render_template("./index.html")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload")
def upload_file():
    return render_template("upload.html")


@app.route("/uploader", methods=["GET", "POST"])
def uploader():
    if request.method == "POST":
        f = request.files["file"]
        f.save(secure_filename(f.filename))
        return "file uploaded successfully"


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
