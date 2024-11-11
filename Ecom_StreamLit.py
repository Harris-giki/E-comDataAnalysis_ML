# Importing necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Load and display dataset
@st.cache
def load_data():
    df = pd.read_csv('/ecommerce.csv')  # Make sure to update the path to your dataset
    return df

df = load_data()

# Title of the app
st.title('E-commerce Yearly Amount Spent Prediction')

# Display dataset overview
st.subheader('Dataset Overview')
st.write(df.head())  # Display the first few rows of the dataset
st.write(df.describe())  # Display statistical summary of the dataset

# EDA: Exploratory Data Analysis Visualizations
st.subheader('Exploratory Data Analysis')

# Jointplot for Time on Website vs Yearly Amount Spent
st.write('**Jointplot between Time on Website and Yearly Amount Spent**')
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df, alpha=0.5)
st.pyplot()

# Jointplot for Time on App vs Yearly Amount Spent
st.write('**Jointplot between Time on App and Yearly Amount Spent**')
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df, alpha=0.5)
st.pyplot()

# Pairplot for all features
st.write('**Pairplot for All Features**')
sns.pairplot(df, kind='scatter', plot_kws={'alpha': 0.4})
st.pyplot()

# lmplot for Length of Membership vs Yearly Amount Spent
st.write('**Linear Regression Plot between Length of Membership and Yearly Amount Spent**')
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df, scatter_kws={'alpha': 0.3})
st.pyplot()

# Feature Selection using Correlation Analysis
st.subheader('Feature Selection (Correlation Analysis)')
correlation_matrix = df.corr()
st.write(correlation_matrix)

# Select top features for prediction
top_features = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
X = df[top_features]
y = df['Yearly Amount Spent']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Linear Regression Model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Display model coefficients
st.subheader('Model Coefficients')
coeff_df = pd.DataFrame(lm.coef_, top_features, columns=['Coefficient'])
st.write(coeff_df)

# Making predictions
predictions = lm.predict(X_test)

# Displaying performance metrics
st.subheader('Performance Metrics')

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)

st.write(f"Mean Absolute Error: {mae}")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Root Mean Squared Error: {rmse}")

# Residuals Analysis
st.subheader('Residuals Analysis')
residuals = y_test - predictions
sns.histplot(residuals, bins=50, kde=True)
st.pyplot()

# Allow users to input new data for prediction
st.subheader('Predict Yearly Amount Spent')
avg_session_length = st.number_input('Avg. Session Length (in minutes)', min_value=0.0, value=33.0)
time_on_app = st.number_input('Time on App (in minutes)', min_value=0.0, value=12.0)
time_on_website = st.number_input('Time on Website (in minutes)', min_value=0.0, value=35.0)
length_of_membership = st.number_input('Length of Membership (in years)', min_value=0.0, value=4.0)

# Create a DataFrame for the user input
user_input = pd.DataFrame([[avg_session_length, time_on_app, time_on_website, length_of_membership]], 
                          columns=top_features)

# Predict the output using the trained model
user_prediction = lm.predict(user_input)

# Display the prediction
st.write(f'Predicted Yearly Amount Spent: ${user_prediction[0]:.2f}')
