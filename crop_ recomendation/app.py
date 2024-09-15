from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Correct path to the new model file
model_path = r'C:\Users\Debanjan Mondal\Desktop\crop_ recomendation\model\NBClassifier.pkl'

# Load the new pre-trained model
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Please check the server logs."

    try:
        # Get input data from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pH = float(request.form['pH'])
        rainfall = float(request.form['rainfall'])

        # Prepare the input data
        input_features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])

        # Predict using the model
        prediction = model.predict(input_features)

        # Return the result
        return render_template('index.html', prediction_text=f'The recommended crop is {prediction[0]}')
    
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
