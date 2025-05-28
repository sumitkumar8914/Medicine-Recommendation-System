import streamlit as st
import pickle
import numpy as np




def predicted_value(patient_symptoms):
    input_vector=np.zeros(len(symptom_lists))

    for item in patient_symptoms:
        input_vector[symptom_lists[item]]=1
    return disease_lists[model1.predict([input_vector])[0]]

predicted_value=pickle.load(open('predicted_value.pkl','rb'))

disease_lists=pickle.load(open('disease_lists.pkl','rb'))

symptom_lists=pickle.load(open('symtom_lists.pkl','rb'))
symptom_list=symptom_lists.keys()
model1=pickle.load(open('model1.pkl','rb'))




st.sidebar.title("Health Care")
page = st.sidebar.radio("Select a page", ["Home", "About", "Contact", "Developer", "Vlog"])

st.markdown("""
    <style>
       h1 {
            font-size: 62px; 
            color:blue;
        }
        .st-emotion-cache-jkfxgf p{
        font-size:25px;
        }
        .st-emotion-cache-1rsyhoq p{
        font-size:20px;}
        .st-emotion-cache-1rsyhoq p {
        font-size: 24px;}
        .st-emotion-cache-12h5x7g p {
        font-size: 18px;}
        .st-emotion-cache-1rsyhoq p {
        font-size: 18px;}
        .st-emotion-cache-v2soli {
        min-height: 3.0rem;
        width: 200px;}
        .st-emotion-cache-ue6h4q {
        font-size:20px;}
        .st-emotion-cache-6qob1r {
        width:200px;}
        .st-emotion-cache-1wmy9hl{
    }


""", unsafe_allow_html=True)


if page == "Home":
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image('images.png', width=110)
    with col2:
        st.title("Health Care Solution")


    selected_symptom = st.multiselect(
        'Please Enter Your Symptom (one or more)',
        (symptom_list)
    )

    predicted_disease=None

    if st.button('Predicted Disease',type="primary"):
        predicted_disease=predicted_value(selected_symptom)
        st.success(f"Predicted Disease: {predicted_disease}")


    def helper(Disease):
        desc = description_df[description_df['Disease'] == Disease]['Description']
        desc = ",".join([w for w in desc])

        pre = precaution_df[precaution_df['Disease'] == Disease][
            ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        pre = [column for column in pre.values]

        med = medication_df[medication_df['Disease'] == Disease]['Medication']
        med = [med for med in med.values]

        die = diets_df[diets_df['Disease'] == Disease]['Diet']
        die = [die for die in die.values]

        wrkout = workout_df[workout_df['disease'] == Disease]['workout']

        return desc, pre, med, die, wrkout


    helper = pickle.load(open('helper.pkl', 'rb'))
    description_df = pickle.load(open('description_df.pkl', 'rb'))
    precaution_df = pickle.load(open('precaution_df.pkl', 'rb'))
    medication_df = pickle.load(open('medication_df.pkl', 'rb'))
    diets_df = pickle.load(open('diets_df.pkl', 'rb'))
    workout_df = pickle.load(open('workout_df.pkl', 'rb'))

    if predicted_disease:
        desc, pre, med, die, wrkout = helper(predicted_disease)

        with st.expander("Description"):
            st.write(desc)

        with st.expander("Precaution"):
            i = 1
            for p_i in pre[0]:
                st.write(i, ": ", p_i)
                i += 1

        with st.expander("Medications"):
            for i in med:
                st.write(i)

        with st.expander("Workout"):
            i = 1
            for w_i in wrkout:
                st.write(i, ": ", w_i)
                i += 1

        with st.expander("Diets"):
            i = 1
            for d_i in die:
                st.write(i, ": ", d_i)
                i += 1



elif page == "Developer":
    st.title("Developer")
    st.write("""
        This app was developed by a team of healthcare and AI experts. 
        The goal is to assist patients in identifying possible diseases 
        and getting recommendations for treatment and care.
    """)

elif page == "Vlog":
    st.title("Vlog")
    st.write("""
        This is the blog page where we share articles about health, medicine, 
        and technology in healthcare. Stay tuned for our latest posts!
    """)


elif page == "Contact":
    st.title("Contact Us")
    st.write("""
        You can reach us at:

        Email: amazingsumit8914@.com  
        Phone: 6205585240
    """)

elif page == "About":
    st.title("About Us")
    st.write("""
        Welcome to the Medicine Recommendation System! This app uses machine learning to 
        predict possible diseases based on your symptoms. It also provides description 
        Precaution medication diets and workout recommendations  
    """)