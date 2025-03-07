import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st

# Streamlit app title
st.title("Welcome to the Most Accurate Prediction App!")

# Streamlit file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    try:
        # Load the dataset
        df = pd.read_csv(uploaded_file)

        # Allow the user to select columns to keep
        st.subheader("Step 1: Select Columns to Keep")
        columns_to_keep = st.multiselect("Select columns to keep", df.columns.tolist(), default=df.columns.tolist())
        
        if not columns_to_keep:
            st.error("Error: Please select at least one column to keep.")
        else:
            df = df[columns_to_keep]
            st.write("Data after filtering columns:")
            st.write(df.head())

            # Allow the user to select target columns
            st.subheader("Step 2: Select Target Column(s) to Predict")
            target_columns = st.multiselect("Select target column(s) to predict", df.columns.tolist())

            if not target_columns:
                st.error("Error: Please select at least one target column.")
            else:
                # Preprocessing for numerical and categorical features
                numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

                # Remove target columns from feature lists
                numerical_features = [col for col in numerical_features if col not in target_columns]
                categorical_features = [col for col in categorical_features if col not in target_columns]

                if not numerical_features and not categorical_features:
                    st.error("Error: No features left for training after removing target columns. Please select different target columns.")
                else:
                    # Preprocessing for numerical data (impute missing values with mean and scale)
                    numerical_transformer = Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler())
                    ])

                    # Preprocessing for categorical data (impute missing values with most frequent and one-hot encode)
                    categorical_transformer = Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ])

                    # Combine preprocessing steps
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", numerical_transformer, numerical_features),
                            ("cat", categorical_transformer, categorical_features)
                        ])

                    # Train a model for each target column
                    models = {}
                    for target_column in target_columns:
                        st.subheader(f"Training model for target column: {target_column}")

                        # Separate features (X) and target (y)
                        X = df.drop(columns=target_columns)
                        y = df[target_column]

                        # Check if the target column is numeric or categorical
                        if pd.api.types.is_numeric_dtype(y):
                            st.write(f"{target_column} is numeric. Using RandomForestRegressor.")
                            model = Pipeline(steps=[
                                ("preprocessor", preprocessor),
                                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
                            ])
                        else:
                            st.write(f"{target_column} is categorical. Using RandomForestClassifier.")
                            model = Pipeline(steps=[
                                ("preprocessor", preprocessor),
                                ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
                            ])

                        # Split the data into training and testing sets
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Train the model
                        model.fit(X_train, y_train)

                        # Make predictions on the test set
                        y_pred = model.predict(X_test)

                        # Evaluate the model
                        if pd.api.types.is_numeric_dtype(y):
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            st.write(f"Mean Squared Error: {mse}")
                            st.write(f"R² Score: {r2}")
                        else:
                            accuracy = accuracy_score(y_test, y_pred)
                            st.write(f"Accuracy: {accuracy}")

                        # Save the model for later prediction
                        models[target_column] = model

                    # Predict for new data
                    st.subheader("Step 3: Predict for New Data")
                    new_data = {}
                    for feature in numerical_features + categorical_features:
                        if feature in numerical_features:
                            new_data[feature] = st.number_input(f"Enter {feature}", value=0)
                        else:
                            new_data[feature] = st.text_input(f"Enter {feature}", value="")

                    if st.button("Predict"):
                        new_data_df = pd.DataFrame([new_data])
                        predictions = {}
                        for target_column, model in models.items():
                            try:
                                predicted_value = model.predict(new_data_df)
                                predictions[target_column] = predicted_value[0]
                            except Exception as e:
                                st.error(f"Error predicting {target_column}: {e}")
                                predictions[target_column] = "Error"

                        # Display predictions in a table
                        st.subheader("Predictions")
                        predictions_df = pd.DataFrame(predictions, index=[0])
                        st.dataframe(predictions_df)  # Use st.dataframe() for an interactive table
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Waiting for file to upload...!")