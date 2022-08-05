from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import random as rand
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__)
CORS(app)

@app.route("/getPredictions")
def get_predictions():
    model = XGBClassifier()
    model.load_model("saved_models/xgb.json")
    # get random row from test dataset
    rand_int = rand.randint(0, len(test)-1)
    row_to_predict = np.array([test.iloc[rand_int]])
    prediction = model.predict(row_to_predict)
    return jsonify(
        {
            "code": 200,
            "message": "Will be bankrupt" if prediction == 1 else "Will not be bankrupt"
        }
    )

if __name__ == "__main__":
    test = pd.read_csv("datasets/bankruptcy_Test_X.csv")
    test = test.drop(["ID"], axis=1)
    app.run(host="0.0.0.0", port=5000, debug=True)