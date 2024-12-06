import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Game Introduction
st.title("Data Detective Hunt: Crack the Code!")
st.markdown("""
Welcome, Detective! 🕵️‍♂️ Your mission is to uncover the team with the most wins in 2019 
and use predictive analytics to solve the case. Follow the tasks and prove your data 
analysis skills!
""")

# Sidebar for Navigation
st.sidebar.header("Navigation")
step = st.sidebar.radio(
    "Choose your task:",
    ["Introduction", "Task 1: Upload Dataset", "Task 2: Clean the Data", 
     "Task 3: Visualize Data", "Task 4: Build Predictive Model", "Final Task"]
)

# Dataset Storage
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None

# Task 1: Upload the Dataset
if step == "Task 1: Upload Dataset":
    st.header("Task 1: Upload the Dataset")
    uploaded_file = st.file_uploader("Upload the Basketball Data CSV", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_data = df
        st.success("Dataset uploaded successfully! Here's a preview:")
        st.write(df.head())

        st.info("Proceed to Task 2 to clean the data.")
    else:
        st.warning("Please upload a dataset to continue.")

# Task 2: Clean the Data
elif step == "Task 2: Clean the Data":
    st.header("Task 2: Clean the Data")

    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data

        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values)

        if missing_values.sum() > 0:
            st.warning("Your dataset contains missing values. Choose how to handle them:")
            action = st.selectbox("Select an action:", 
                                  ["Remove rows with missing values", "Fill missing values with mean"])
            if st.button("Apply"):
                if action == "Remove rows with missing values":
                    df = df.dropna()
                    st.success("Rows with missing values removed!")
                elif action == "Fill missing values with mean":
                    df = df.fillna(df.mean())
                    st.success("Missing values filled with column mean!")
                
                st.session_state.cleaned_data = df
                st.write("Cleaned Dataset:")
                st.write(df.head())
                st.info("Proceed to Task 3 to explore and visualize the data.")
        else:
            st.success("No missing values found! Proceed to Task 3.")
            st.session_state.cleaned_data = df
    else:
        st.warning("Please upload a dataset in Task 1 first.")
     
# Task 3: Visualize Data
elif step == "Task 3: Visualize Data":
    st.header("Task 3: Visualize Data")

    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data

        # Filter data for 2019
        if 'Year' in df.columns:
            df_2019 = df[df['Year'] == 2019]
            st.subheader("Filtered Data for 2019")
            st.write(df_2019)

            # Visualization
            if 'Team' in df.columns and 'Games_Won' in df.columns:
                st.subheader("Team Performance in 2019")
                team_stats = df_2019.groupby('Team')['Games_Won'].sum().sort_values(ascending=False)
                fig = px.bar(team_stats, x=team_stats.index, y=team_stats.values, title="Games Won by Teams")
                st.plotly_chart(fig)
            else:
                st.warning("The dataset must contain 'Team' and 'Games_Won' columns.")
        else:
            st.warning("Guide: The visualized data must contain 'Team, Year, and Games Won'.")
    else:
        st.warning("Please clean the dataset in Task 2 first.")

# Task 4: Build Predictive Model
elif step == "Task 4: Build Predictive Model":
    st.header("Task 4: Build a Predictive Model")

    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data

        if 'Games_Won' in df.columns and 'Points_Scored' in df.columns:
            st.subheader("Predicting Games Won Based on Points Scored")

            # Prepare data
            X = df[['Points_Scored']]
            y = df['Games_Won']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions and evaluation
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error of the model: {mse:.2f}")

            # Plot predictions
            fig, ax = plt.subplots()
            ax.scatter(X_test, y_test, color='blue', label='Actual')
            ax.plot(X_test, y_pred, color='red', label='Predicted')
            ax.set_title("Games Won Prediction")
            ax.set_xlabel("Points Scored")
            ax.set_ylabel("Games Won")
            ax.legend()
            st.pyplot(fig)

            st.info("Proceed to the Final Task to solve the mystery!")
        else:
            st.warning("Guide: The dataset must contain 'Games_Won' and 'Points_Scored' columns for modeling.")
    else:
        st.warning("Please clean the dataset in Task 2 first.")

# Final Task: Solve the Mystery
elif step == "Final Task":
    st.header("Final Task: Solve the Mystery")

    if st.session_state.cleaned_data is not None:
        st.text("Based on your analysis, which team won the most games in 2019?")
        team_answer = st.text_input("Enter your answer:")

        if team_answer.strip().lower() == "virginia":
            st.success("Correct! Virginia won the 2019 NCAA Championship.")
            st.balloons()
        elif team_answer:
            st.error("Incorrect! Try again.")
    else:
        st.warning("Please complete the previous tasks first.")
