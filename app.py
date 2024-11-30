# pip install flask
from datetime import datetime
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np


# Loading the mlr model
model = pickle.load(open('model.pkl', 'rb'))

# Flask is used for creating your application
# render template is used for rendering the HTML page
app = Flask(__name__)  # your application


@app.route('/')  # default route
def home():
    return render_template('index.html')  # rendering your home page.


@app.route('/pred', methods=['POST'])  # prediction route
def predict1():
    '''
    For rendering results on HTML 
    '''
    
    date_of_journey = request.form.get('DOJ')
    day_of_journey = request.form.get('Day')
    flight_code = int(request.form.get('flightcode'))
    class_of_ticket = int(request.form.get('class'))
    source_airport = int(request.form.get('sourceAirport'))
    destination_airport = int(request.form.get('destinationAirport'))
    total_stops = int(request.form.get('totalStops'))
    departure_time = int(request.form.get('departureTime'))
    arrival_time = int(request.form.get('arrivalTime'))
    duration = int(request.form.get('duration'))
    daysleft= int(request.form.get('daysleft'))

    # Extracting day, month, and year from the date of journey
    date_of_journey = datetime.strptime(date_of_journey, '%Y-%m-%d')
    day_of_journey1 = date_of_journey.day
    month_of_journey = date_of_journey.month
    year_of_journey = date_of_journey.year



    # Create a DataFrame from the form data
    input_data = [
        day_of_journey,
        flight_code,
        class_of_ticket,
        source_airport,
        departure_time,
        total_stops,
        arrival_time,
        destination_airport,
        duration,
        daysleft,
        day_of_journey1,
        month_of_journey,
        year_of_journey
    ]

    input_df = pd.DataFrame([input_data],columns=None)

    # Make prediction using the pre-trained model
    prediction = model.predict(input_df)
    print(prediction)

    return render_template("index.html", result="The predicted fare is " + str(np.round(prediction[0])))


# running your application
if __name__ == "__main__":
    app.run()
