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

  

    if filtered_df.empty:
        st.warning("No data matches the selected filter.")
    else:

        st.subheader("Plot Data")
        x_columns = st.selectbox("Select x-axis column", columns)
        y_columns = st.selectbox("Select y-axis column", columns)
        if (x_columns == y_columns):
            if st.button("Generate Plot"):
                if x_columns in filtered_df.columns and y_columns in filtered_df.columns :
                    if pd.api.types.is_numeric_dtype(filtered_df[y_columns]):
                        st.line_chart(filtered_df.set_index(x_columns)[y_columns])
                    else:
                        st.error(f"Error: The selected column '{y_columns}' must contain numeric data.")
                else:
                    st.error(f"Error: The selected column '{y_columns}' does not exist in the filtered data.")
        else:
            st.error(f"Error:choose both x-axis and y-axis different" )
else:
    st.write("Wainting for file to upload...!") 
