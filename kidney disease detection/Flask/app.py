#importing the necessary dependencies
import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle
import os

app=Flask(__name__,template_folder='templates') #iniatialize a Flask app

model= pickle.load(open('CKD.pkl','rb'))

@app.route('/')#route to display home page
def home():
    return render_template('home.html')

@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template("index.html")

@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/Predict',methods=['POST']) #route to show prediction in web UI
def predict():

    #reading the inputs given by the user
    input_features =[float(x) for x in request.form.values()]
    features_value =[np.array(input_features)]
    
    features_name =['blood_urea','blood glucose random','anemia',
                    'coronary_artery_disease','pus_cell','red_blood_cells'
                    'diabetesmellitus','pedal_edema']
    
    df=pd.DataFrame(features_value,columns=features_name)
    
    #prediction using the loaded model file
    output=model.predict(df)
    
    #showing the prediction results in UI
    return render_template('result.html',prediction_text=output)

if __name__=='__main__' :
    #running the app
    app.run(debug=True)

