from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

model = pickle.load(open("happydata.pkl", "rb"))
scaler = pickle.load(open("happydata_sc.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/home')
def home1():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/predict')
def courses():
    return render_template("courses.html")


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        infoavail = float(request.form["infoavail"])
        housecost = float(request.form["housecost"])
        schoolquality = float(request.form["schoolquality"])
        policetrust = float(request.form["policetrust"])
        streetquality = float(request.form["streetquality"])
        ëvents = float(request.form["ëvents"])

        # Use the scaler to transform the input data
        data = np.array([[infoavail, housecost, schoolquality, policetrust, streetquality, ëvents]])

        # Predict using the model
        pred = model.predict(data)
        pred = int(pred[0])

        if pred == 0:
            return render_template("events.html", predict="Unhappy")
        else:
            return render_template("events.html", predict="Happy")

if __name__ == "__main__":
    app.run(debug=True, port=4444)