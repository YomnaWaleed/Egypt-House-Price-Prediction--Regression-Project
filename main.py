from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the dataset and model
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", "rb"))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    house_types = sorted(data['House_Type'].unique())  # Assuming you have this in your dataset
    floors = sorted(data['Floor'].unique())  # Assuming you have this in your dataset
    return render_template('index.html', locations=locations, house_types=house_types, floors=floors)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    total_sqft = request.form.get('total_sqft')
    house_type = request.form.get('house_type')  # Collecting 'House_Type' from form input
    floor = request.form.get('floor')  # Collecting 'Floor' from form input

    # Create the DataFrame with all necessary columns
    input_df = pd.DataFrame([[location, total_sqft, bhk, house_type, floor]],
                            columns=['location', 'Size', 'BHK', 'House_Type', 'Floor'])

    try:
        prediction = pipe.predict(input_df)[0]
        return f"Prediction: {prediction}"
    except KeyError as e:
        return f"Prediction: KeyError: {e}. Please check your input data."

if __name__ == "__main__":
    app.run(debug=True, port=5001)
