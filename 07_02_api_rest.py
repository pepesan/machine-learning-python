from flask import Flask, request
import pandas as pd
import json
import pickle

with open("iris_knn_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)


app = Flask(__name__)
target_names = ['setosa', 'versicolor', 'virginica']
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = pd.DataFrame(data['features'])  # Convertir a DataFrame
    print(features)
    prediction = loaded_model.predict(features)[0].tolist() # Predecir con KNN
    print(prediction)
    print(target_names[prediction])
    diccionario = {
        'prediction': prediction,
        'target': target_names[prediction]
    }
    json_string = json.dumps(diccionario, indent=4)
    return json_string

if __name__ == '__main__':
    app.run(debug=True)

