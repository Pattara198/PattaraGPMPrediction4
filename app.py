from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

import numpy as np  
from tensorflow.keras.models import load_model
import joblib



def return_prediction(model,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    
    s_Cylinders = sample_json['cylinders_']
    s_Displacement = sample_json['displacement_']
    s_Horsepower = sample_json['horsepower_']
    s_Weight = sample_json['weight_']
    s_Acceleration = sample_json['acceleration_']
    s_Year = sample_json['year_']
    s_Origin = sample_json['origin_']    
    
    X = [[s_Cylinders,s_Displacement,s_Horsepower,s_Weight,s_Acceleration,s_Year,s_Origin]]
    
    prediction = model.predict(X)
    
    
    return prediction



app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
flower_model = load_model("Pattara_Model.h5")


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class FlowerForm(FlaskForm):
    Cylinders_s = TextField('Cylinders_Info')
    Displacement_s = TextField('Displacement_Info')
    Horsepower_s = TextField('Horsepower_Info')
    Weight_s = TextField('Weight_Info')
    Acceleration_s = TextField('Acceleration_Info')
    Year_s = TextField('Year_Info')
    Origin_s = TextField('Origin_Info')


    submit = SubmitField('Analyze')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = FlowerForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['Cylinders_s'] = form.Cylinders_s.data
        session['Displacement_s'] = form.Displacement_s.data
        session['Horsepower_s'] = form.Horsepower_s.data
        session['Weight_s'] = form.Weight_s.data
        session['Acceleration_s'] = form.Acceleration_s.data
        session['Year_s'] = form.Year_s.data
        session['Origin_s'] = form.Origin_s.data



        return redirect(url_for("prediction"))


    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['cylinders_'] = int(session['Cylinders_s'])
    content['displacement_'] = float(session['Displacement_s'])
    content['horsepower_'] = float(session['Horsepower_s'])
    content['weight_'] = int(session['Weight_s'])
    content['acceleration_'] = float(session['Acceleration_s'])
    content['year_'] = int(session['Year_s'])
    content['origin_'] = int(session['Origin_s'])
    
    

    results = return_prediction(model=flower_model,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)
