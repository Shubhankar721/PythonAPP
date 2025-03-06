import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 

st.title("Simple Data Dashaboard")

uploaded_file = st.file_uploader("Choose a CSV file",type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Privew")
    st.write(df.head())

    st.subheader('Data Summary')
    st.write(df.describe())

    st.subheader('Filter Data')
    columns = df.columns.tolist()
    selected_columns = st.selectbox("Select colum  to filter by ",columns)
    unique_values = df[selected_columns].unique()
    selected_values = st.selectbox("Select value", unique_values) 

    filtered_df = df[df[selected_columns] == selected_values]
    st.write(filtered_df)

    st.subheader("Plot data")
    x_columns = st.selectbox("Select x-axis column",columns)
    y_columns = st.selectbox("Select y-axis column",columns)

    if st.button("Generate Plot"):
        st.line_chart(filtered_df.set_index(x_columns)[y_columns])
else:
    st.write("Wainting for fileto upload...!") 