import pickle
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
   dv, model = pickle.load(f_in)

app = Flask('Heart_disease')

@app.route('/predict', methods=['POST'])
def predict():
   patient = request.get_json()
   X = dv.transform([patient])
   y_pred = model.predict(X)

   results = {
      'Disease_probability': float(y_pred)
   }

   return jsonify(results)

if __name__ == "__main__":
   app.run(debug=True, host='0.0.0.0', port=6969)
   