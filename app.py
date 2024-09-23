from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Iris Prediction Model API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = scaler.transform([[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]])
    prediction = model.predict(features)
    return jsonify({'species': iris.target_names[prediction][0]})

if __name__ == '__main__':
    app.run(debug=True)
