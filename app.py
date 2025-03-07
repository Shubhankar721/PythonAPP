
import streamlit as st
import pandas as pd

# Title of the app
st.title("Simple Data Dashboard with Data Cleaning")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Data Cleaning Section
        st.subheader("Data Cleaning")

        # Display original data
        st.write("Original Data:")
        st.write(df.head())

        # Replace null values
        st.write("### Step 1: Replace Null Values")
        if df.isnull().sum().sum() > 0:
            st.write("Missing values detected:")
            st.write(df.isnull().sum())

            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.write("Data after replacing null values:")
            st.write(df.head())
        else:
            st.write("No missing values found.")

        # Remove Duplicates
        st.write("### Step 2: Remove Duplicates")
        if df.duplicated().sum() > 0:
            st.write(f"Number of duplicate rows: {df.duplicated().sum()}")
            if st.checkbox("Remove duplicate rows"):
                df = df.drop_duplicates()
                st.write("Data after removing duplicates:")
                st.write(df.head())
        else:
            st.write("No duplicate rows found.")

        # Filter Irrelevant Data
        st.write("### Step 3: Filter Irrelevant Data")
        columns_to_keep = st.multiselect("Select columns to keep", df.columns.tolist(), default=df.columns.tolist())
        df = df[columns_to_keep]
        st.write("Data after filtering columns:")
        st.write(df.head())

        # Data Preview
        st.subheader("Cleaned Data Preview")
        st.write(df.head())

        # Data Summary
        st.subheader('Data Summary')
        st.write(df.describe())

        # Filter Data
        st.subheader('Filter Data')
        columns = df.columns.tolist()
        selected_columns = st.selectbox("Select column to filter by", columns)
        unique_values = df[selected_columns].unique()
        selected_values = st.selectbox("Select value", unique_values)

        # Apply filter
        filtered_df = df[df[selected_columns] == selected_values]
        st.write(filtered_df)

        # Check if filtered DataFrame is empty
        if filtered_df.empty:
            st.warning("No data matches the selected filter.")
        else:
            # Plot Data
            st.subheader("Plot Data")
            x_columns = st.selectbox("Select x-axis column", columns)
            y_columns = st.selectbox("Select y-axis column", columns)

            if st.button("Generate Plot"):
                # Check if selected columns exist in the filtered DataFrame
                if x_columns in filtered_df.columns and y_columns in filtered_df.columns:
                    # Check if y_columns contains numeric data
                    if pd.api.types.is_numeric_dtype(filtered_df[y_columns]):
                        st.line_chart(filtered_df.set_index(x_columns)[y_columns])
                    else:
                        st.error(f"Error: The selected column '{y_columns}' must contain numeric data.")
                else:
                    st.error(f"Error: The selected column '{y_columns}' does not exist in the filtered data.")

        # Add a download button for the cleaned DataFrame
        if not df.empty:
            st.subheader("Download Cleaned Data")
            # Convert the cleaned DataFrame to a CSV file
            csv = df.to_csv(index=False).encode('utf-8')
            # Add a download button
            st.download_button(
                label="Download Cleaned Data as CSV",
                data=csv,
                file_name='cleaned_data.csv',
                mime='text/csv',
            )
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.write("Waiting for file to upload...!")

    st.subheader("Demo CSV File")
    st.write("Don't have a CSV file? Download our demo file to test the app:")

    demo_data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 30, None, 40, 45],
    "Salary": [50000, None, 70000, 80000, 90000],
    "Join Date": ["2023-01-01", "2023-02-15", "2023-03-10", None, "2023-05-20"]
    }
    demo_df = pd.DataFrame(demo_data)

    # Convert the demo DataFrame to a CSV file
    demo_csv = demo_df.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="Download Demo CSV",
    data=demo_csv,
    file_name='demo_data.csv',
    mime='text/csv',
    )