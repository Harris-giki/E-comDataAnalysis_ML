import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Loading the dataset
df = pd.read_csv('ecommerce.csv')

# Features and target
x = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train the model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Streamlit UI for user input
st.title('E-commerce Yearly Spend Prediction')

st.write("Enter the details below:")
st.write("Each feature is measured in the unit of 'Minutes' of time")
# User input fields
avg_session_length = st.number_input('Avg. Session Length', min_value=0.0, value=33.0, step=0.1)
time_on_app = st.number_input('Time on App', min_value=0.0, value=12.0, step=0.1)
time_on_website = st.number_input('Time on Website', min_value=0.0, value=37.0, step=0.1)
length_of_membership = st.number_input('Length of Membership', min_value=0.0, value=3.5, step=0.1)

# Predict button
if st.button('Predict Yearly Spend'):
    # Making prediction using the model
    input_data = pd.DataFrame([[avg_session_length, time_on_app, time_on_website, length_of_membership]], 
                              columns=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'])
    prediction = lm.predict(input_data)

    # Display the prediction
    st.write(f"Predicted Yearly Spend: ${prediction[0]:.2f}")

# Optionally, add some explanation or more details about how the model works
st.write("""
    This model predicts the yearly amount spent by customers based on the following features:
    - Average Session Length
    - Time spent on the App
    - Time spent on the Website
    - Length of Membership
    
    The model was trained using historical data of e-commerce users and their spending behavior.
    The prediction is based on the inputs you provide above.
""")
