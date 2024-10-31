from flask import Flask, request , render_template 
import pandas as pd
import numpy as np
import pickle
app=Flask(__name__)
model = pickle.load(open("LinearRegressionModel.pkl","rb"))
car = pd.read_csv("Cleaned_Car.csv")
@app.route("/")
def index(): 
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    companies.insert(0,"Select a Company")
    year = sorted(car["year"].unique(),reverse=True)
    fuel_type= car["fuel_type"].unique()
    return render_template("index.html",companies=companies ,car_models=car_models,years=year,fuel_type=fuel_type)
@app.route("/predict",methods=["POST","GET"])
def predict(): 
    company= request.form["company"]
    car_model=request.form["car_models"]
    year=int(request.form["year"])
    fuel_type=request.form["fuel_type"]
    kilometers=int(request.form["kilo_driven"])
    prediction = model.predict(pd.DataFrame([[car_model,company,year,kilometers,fuel_type]],columns=["name","company","year","kms_driven","fuel_type"]))
    return str(np.round(prediction[0],2))
if __name__ == "__main__": 
    app.run(debug=True,port=5001) 