from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table
import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime


import joblib
pipe_lr = joblib.load(open("emotion_classifier_pipe_lr.pkl", "rb"))


def predire_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


emotions_emoji_dict = {"anger": "😠", "fear": "😨😱",
                       "joy": "😂", "sadness": "😔", "surprise": "😮"}


# Main Application
def main():
    st.title("Classificateur d'émotions")
    menu = ["Acceuil", "Moniteur", "A propos"]
    choix = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()
    if choix == "Acceuil":
        add_page_visited_details("Acceuil", datetime.now())
        st.subheader("Acceuil Emotion Dans Texte")
        with st.form(key='emotion_clf_form'):
            text = st.text_area("Tapez votre texte ici")
            soum_text = st.form_submit_button(label='Soumettre')
        if soum_text:
            col1, col2 = st.beta_columns(2)
            prediction = predire_emotions(text)
            probabilites = get_prediction_proba(text)
            add_prediction_details(text, prediction, np.max(
                probabilites), datetime.now())
            with col1:
                st.success("Text entré")
                st.write(text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confiance:{}".format(np.max(probabilites)))
            with col2:
                st.success("Prediction Probabilités")
                proba_df = pd.DataFrame(probabilites, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probabilite"]
                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions', y='probabilite', color='emotions')
                st.altair_chart(fig, use_container_width=True)
    elif choix == "Moniteur":
        add_page_visited_details("Moniteur", datetime.now())
        st.subheader("Moniteur App")
        with st.beta_expander("Metriques"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=[
                                                'Pagename', 'Date_De_Visite'])
            st.dataframe(page_visited_details)
            pg_count = page_visited_details['Pagename'].value_counts(
            ).rename_axis('Pagename').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(
                x='Pagename', y='Counts', color='Pagename')
            st.altair_chart(c, use_container_width=True)
            p = px.pie(pg_count, values='Counts', names='Pagename')
            st.plotly_chart(p, use_container_width=True)
        with st.beta_expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=[
                                       'Text', 'Prediction', 'Probabilité', 'Date_de_visite'])
            st.dataframe(df_emotions)
            prediction_count = df_emotions['Prediction'].value_counts(
            ).rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(
                x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)
    else:
        st.subheader("A propos")
        add_page_visited_details("A propos", datetime.now())


if __name__ == '__main__':
    main()
