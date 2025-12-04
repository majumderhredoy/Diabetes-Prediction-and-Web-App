from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load('diabetes_model.pkl')


gender_map = {'Male':0, 'Female': 1, 'Other': 2}
smoking_map = {'never': 0, 'former': 1, 'current': 2, 'not current': 3, 'ever':4}

@app.route('/', methods=['GET'])
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
   gender = request.form['gender']
   age= float(request.form['age'])
   hypertension = int(request.form['hypertension'])
   heart_disease = int(request.form['heart_disease'])
   smoking_history = request.form['smoking_history']
   bmi = float(request.form['bmi'])
   hba1c = float(request.form['hba1c'])
   blood_glucose = float(request.form['blood_glucose'])


   #Convert  to model input
   features = np.array([
      gender_map[gender],
      age,
      hypertension,
      heart_disease,
      smoking_map[smoking_history],
      bmi,
      hba1c,
      blood_glucose]).reshape(1, -1)

   prediction = model.predict(features)[0]

   result = 'Diabetes Detected' if prediction== 1 else 'No Diabetes'
   return render_template('form.html', prediction=result)
if __name__ == "__main__":
   app.run(debug=True)
