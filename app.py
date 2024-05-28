import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

ridge_model=pickle.load(open('model/flite_Ridge.pkl','rb'))
standard_scaler=pickle.load(open('model/flite_scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/',methods=['GET','POST'])
def Predict_datapoint():
    if request.method=='POST':
        Airline=float(request.form.get('Airline'))
        Date_of_Journey=float(request.form.get('Date_of_Journey'))
        Source=float(request.form.get('Source'))
        Destination=float(request.form.get('Destination'))
        Dep_Time=float(request.form.get('Dep_Time'))
        Arrival_Time=float(request.form.get('Arrival_Time'))
        Total_Stops=float(request.form.get('Total_Stops'))

        new_data_scaled=standard_scaler.transform([[Airline,Date_of_Journey,Source,Destination,Dep_Time,Arrival_Time,Total_Stops]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('Home.html',result=result[0])
            
    else:
        return render_template('Home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")
