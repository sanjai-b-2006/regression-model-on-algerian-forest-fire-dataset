import pickle
from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load all your models and the standard scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
linear_model = pickle.load(open('models/linear.pkl', 'rb'))
lasso_model = pickle.load(open('models/lasso.pkl', 'rb'))
elasticnet_model = pickle.load(open('models/elasticnet.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        model_name = request.form.get('model')
        print(f"Selected model: {model_name}")

        if model_name == 'ridge':
            model = ridge_model
        elif model_name == 'linear':
            model = linear_model
        elif model_name == 'lasso':
            model = lasso_model
        elif model_name == 'elasticnet':
            model = elasticnet_model
        else:
            print("Invalid model selected")
            return "Invalid model selected", 400

        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            print("Data received:", Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region)
        except ValueError as e:
            print(f"Error parsing form data: {e}")
            return "Error parsing form data", 400

        new_data_scaled = standard_scaler.transform(np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]))
        result = model.predict(new_data_scaled)

        return render_template('home.html', result=result[0], model_name=model_name)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
