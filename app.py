import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk

model = pk.load(open("Heart_diease_model.pkl", "rb"))

data = pd.read_csv('heart_disease.csv')

st.header('CardioCare ❤')

gender = st.selectbox('Choose Gender', data['Gender'].unique())
if gender == 'Male':
    gen = 1
else:
    gen = 0

age = st.number_input("Enter Your Age Here:")
currentSmoker = st.number_input("Is patient currentSmoker")
cigsPerDay = st.number_input("Enter the No. of Cigarretes per day")
BPMeds = st.number_input("Is patient is on any BP Medicines")
prevalentStroke = st.number_input("Is patient had stroke earlier")
prevalentHyp = st.number_input("Is patient had Hypertension earlier")
diabetes = st.number_input("Diabitis status:")
totChol = st.number_input("Enter Your total Cholestrole:")
sysBP = st.number_input("Enter Your systolic blood pressure:")
diaBP = st.number_input("Enter Your  diastolic blood pressure:")
BMI = st.number_input("Enter Your Body Mass Index:")
heartRate = st.number_input("Enter Your Heart Rate:")
glucose = st.number_input("Enter Your Glucose:")

if st.button('predict'):
            input = np.array([[gen,age,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose]])

            output = model.predict(input)
            if output[0] == 0:
                   stn = "Don't worry the patient do not have any heart disease"
            else:
                   stn = "The patient may have heart disease, it's better to consult a good cardiologist as i am just a ML model"   
            st.markdown(stn)
