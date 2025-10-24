from flask import Flask, jsonify
from flask import request
import pickle

model_file = r"..\model_C1.0.bin"
with open(model_file, 'rb') as f:
    dv, model = pickle.load(f)

app = Flask('churn_app')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json(force=True)
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5

    result = {'churn': bool(churn),
              'probability':y_pred}
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)