from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
df = pd.read_csv(r"C:\Users\khare\Ml_Projects\ML projects\House_prediction\Cleaned_data.csv")
pipe = pickle.load(open(r"C:\Users\khare\Ml_Projects\ML projects\House_prediction\RidgeModel.pkl", "rb"))

@app.route('/')
def index(): 
    location = sorted(df["location"].unique())
    return render_template('index.html', location=location)

@app.route('/predict', methods=["POST"])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bathrooms = request.form.get('bathrooms')
    sqft = request.form.get('sqft')
    
    print(location, bhk, bathrooms, sqft)  # Debugging output to console

    # Creating DataFrame for prediction input
    input_data = pd.DataFrame([[location, sqft, bathrooms, bhk]], columns=["location", "total_sqft", "bath", "BHk"])
    
    # Predicting the price
    prediction = pipe.predict(input_data)[0] * 1e5  # Adjusting for prediction scale
    
    # Return rounded prediction as string
    return str(np.round(prediction, 2))

if __name__ == "__main__": 
    app.run(debug=True, port=5001)
