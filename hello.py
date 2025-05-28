#import libreries and model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

#load dataset
df=pd.read_csv(r"C:\Users\Ayush\Downloads\Training.csv")
df.head(3)

df.shape

df.info()

df.isnull().sum()

X=df.drop('prognosis',axis=1)
y=df['prognosis']

labelE=LabelEncoder()

labelE.fit(y)

y=labelE.transform(y)
y

X.head()

y

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

model1=SVC(kernel='linear')
model1.fit(X_train,y_train)

y_pred=model1.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
accuracy

model2=RandomForestClassifier(n_estimators=500)
model2.fit(X_train,y_train)

y_pred=model2.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
accuracy

model3=MultinomialNB()
model3.fit(X_train,y_train)

y_pred=model3.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
accuracy

print("predicted=",model1.predict(X_test.iloc[2].values.reshape(1,-1)))
print("Actual=",y_test[2])

#save the model
import pickle
pickle.dump(model1,open("model1.pkl",'wb'))

#load database
symtom_df=pd.read_csv(r'C:\Users\Ayush\Downloads\symtoms_df.csv')
symtom_df.head()

precaution_df=pd.read_csv(r'C:\Users\Ayush\Downloads\precautions_df.csv')
precaution_df.head()

workout_df=pd.read_csv(r'C:\Users\Ayush\Downloads\workout_df.csv')
workout_df.head()

description_df=pd.read_csv(r'C:\Users\Ayush\Downloads\description.csv')
description_df.head(2)

diets_df=pd.read_csv(r'C:\Users\Ayush\Downloads\diets.csv')
diets_df.head()

medication_df=pd.read_csv(r"C:\Users\Ayush\Downloads\medications.csv")
medication_df.head(5)

def helper(Disease):
    desc=description_df[description_df['Disease']==Disease]['Description']
    desc=",".join([w for w in desc])

    pre=precaution_df[precaution_df['Disease']==Disease][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre = [column for column in pre.values]
    
    med=medication_df[medication_df['Disease']==Disease]['Medication']
    med=[med for med in med.values]

    die=diets_df[diets_df['Disease']==Disease]['Diet']
    die=[die for die in die.values]

    wrkout=workout_df[workout_df['disease']==Disease]['workout']
    
    return desc,pre,med,die,wrkout

pickle.dump(helper,open("helper.pkl",'wb'))

pickle.dump(description_df,open("description_df.pkl",'wb'))
pickle.dump(precaution_df,open("precaution_df.pkl",'wb'))
pickle.dump(medication_df,open("medication_df.pkl",'wb'))
pickle.dump(diets_df,open("diets_df.pkl",'wb'))
pickle.dump(workout_df,open("workout_df.pkl",'wb'))

# model prediction function
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5,
                 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11,
                 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 
                 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 
                 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26,
                 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
                 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
                 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 
                 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 
                 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59,
                 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 
                 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
                 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73,
                 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 
                 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 
                 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
                 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,'foul_smell_of urine': 90, 
                 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 
                 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 
                 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 
                 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111,
                 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
                 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
                 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
                 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

pickle.dump(symptoms_dict,open("symtom_lists.pkl",'wb'))

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae',
                 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7:
                 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37:
                 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins',
                 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
                 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

pickle.dump(diseases_list,open("disease_lists.pkl",'wb'))

#model prediction function
def predicted_value(patient_symptoms):
    input_vector=np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        input_vector[symptoms_dict[item]]=1
    return diseases_list[model1.predict([input_vector])[0]]

symptoms = input("Enter your symptoms.......")
user_symptoms = [s.strip() for s in symptoms.split(',')]
user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
predicted_disease= predicted_value(user_symptoms)

desc, pre, med, die, wrkout = helper(predicted_disease)

print("=================predicted disease============")
print(predicted_disease)
print("=================description==================")
print(desc)
print("=================precautions==================")
i = 1
for p_i in pre[0]:
    print(i, ": ", p_i)
    i += 1

print("=================medications==================")
for m_i in med:
    print(i, ": ", m_i)
    i += 1

print("=================workout==================")
for w_i in wrkout:
    print(i, ": ", w_i)
    i += 1

print("=================diets==================")
for d_i in die:
    print(i, ": ", d_i)
    i += 1

import sklearn
print(sklearn._version_)

pickle.dump(predicted_value,open("predicted_value.pkl",'wb'))