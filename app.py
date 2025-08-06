from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

#Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Time = request.form.get('time'),
            V1 = request.form.get('v1'),
            V2=request.form.get('v2'),
            V3 = request.form.get('V3'),
            V4 = request.form.get('v4'),
            V5 = request.form.get('v5'),
            V6 = request.form.get('v6'),
            V7 = request.form.get('v7'),
            V8 = request.form.get('v8'),
            V9 = request.form.get('v9'),
            V10 = request.form.get('v10'),
            V11 = request.form.get('v11'),
            V12 = request.form.get('v12'),
            V13 = request.form.get('v13'),
            V14 = request.form.get('v14'),
            V15 = request.form.get('v15'),
            V16 = request.form.get('v16'),
            V17 = request.form.get('v17'),
            V18 = request.form.get('v18'),
            V19 = request.form.get('v19'),
            V20 = request.form.get('v20'),
            V21 = request.form.get('v21'),
            V22 = request.form.get('v22'),
            V23 = request.form.get('v23'),
            V24 = request.form.get('v24'),
            V25 = request.form.get('v25'),
            V26 = request.form.get('v26'),
            V27 = request.form.get('v27'),
            V28 = request.form.get('v28'),
            Amount=request.form.get('amount')
        )
        
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")
        
        predict_pipeline = PredictPipeline()
        print("Mid prediction")
        results = predict_pipeline.predict(pred_df)
        print("After prediction")
        return render_template("home.html", results= results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0")
