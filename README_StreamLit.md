<h1>E-commerce Yearly Spend Prediction</h1>
<h2>Overview</h2>
    <p>This project is an interactive web application built with <strong>Streamlit</strong> and <strong>Scikit-learn</strong> for predicting the <strong>Yearly Amount Spent</strong> by e-commerce users. The app leverages a <strong>Linear Regression model</strong> trained on features like average session length, time spent on the app, time spent on the website, and length of membership. By inputting these features, users can get a predicted yearly spend value for a customer.</p>
    <p>The project provides a simple, intuitive interface that allows users to interact with the model and make predictions based on the provided input values.</p>
    <h2>Project Features</h2>
    <ul>
        <li><strong>Interactive Web Interface</strong>: Built using Streamlit for easy interaction and data entry.</li>
        <li><strong>Prediction Model</strong>: Uses a Linear Regression model to predict the yearly amount spent based on multiple user features.</li>
        <li><strong>User Input</strong>: Users can enter values for session length, time on app, time on website, and membership length to generate predictions.</li>
        <li><strong>Easy to Run</strong>: The app can be executed locally or deployed on a cloud platform with minimal setup.</li>
    </ul>
    <h2>Installation and Setup</h2>
    <h3>Prerequisites</h3>
    <p>Make sure you have the following libraries installed before running the app:</p>
    <pre><code>pip install streamlit pandas scikit-learn</code></pre>
    <h3>Running the Application Locally</h3>
    <ol>
        <li>Clone the repository to your local machine:
            <pre><code>git clone https://github.com/yourusername/ecommerce-prediction-app.git</code></pre>
        </li>
        <li>Navigate into the project folder:
            <pre><code>cd ecommerce-prediction-app</code></pre>
        </li>
        <li>Run the app:
            <pre><code>streamlit run app.py</code></pre>
        </li>
        <li>Open your browser and go to <a href="http://localhost:8501" target="_blank">http://localhost:8501</a> to interact with the app.</li>
    </ol>
    <h3>Input Fields</h3>
    <p>The user is prompted to enter values for the following features:</p>
    <ul>
        <li><strong>Avg. Session Length</strong>: Average session time for the user in minutes.</li>
        <li><strong>Time on App</strong>: Time spent on the app in minutes.</li>
        <li><strong>Time on Website</strong>: Time spent on the website in minutes.</li>
        <li><strong>Length of Membership</strong>: The duration of the user's membership in years.</li>
    </ul>
    <h3>Model Prediction</h3>
    <p>Once the user has entered the required values, they can click the <strong>Predict Yearly Spend</strong> button. The model will output a predicted yearly spend for the user based on the input features.</p>
    <h2>Approach</h2>
    <h3>1. Dataset and Feature Selection</h3>
    <p>The dataset used for this app (assumed to be <code>ecommerce.csv</code>) contains data about e-commerce users, including various user features and their corresponding yearly spend.</p>
    <p><strong>Features Used for Prediction:</strong></p>
    <ul>
        <li><strong>Avg. Session Length</strong></li>
        <li><strong>Time on App</strong></li>
        <li><strong>Time on Website</strong></li>
        <li><strong>Length of Membership</strong></li>
    </ul>
    <p>These features were selected because they have a direct relationship with user engagement, which influences their spending behavior.</p>
    <h3>2. Data Preprocessing</h3>
    <p>The dataset was loaded and checked for missing or outlier values. Features were normalized and scaled appropriately to fit the model.</p>
    <h3>3. Model Selection</h3>
    <p>A <strong>Linear Regression</strong> model was chosen for this project due to its simplicity and effectiveness in predicting continuous outcomes based on a linear relationship between features.</p>
    <pre><code>
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Splitting data into features (X) and target (y)
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
lm = LinearRegression()
lm.fit(X_train, y_train)
    </code></pre>
    <h3>4. Streamlit Interface</h3>
    <p>Streamlit was used to create an interactive web interface. The user can enter their details for the features and get predictions in real-time. The interface was designed to be simple and intuitive for ease of use.</p>
    <pre><code>
import streamlit as st

# Streamlit UI for user input
st.title('E-commerce Yearly Spend Prediction')

st.write("Enter the details below:")

# User input fields
avg_session_length = st.number_input('Avg. Session Length', min_value=0.0, value=33.0, step=0.1)
time_on_app = st.number_input('Time on App', min_value=0.0, value=12.0, step=0.1)
time_on_website = st.number_input('Time on Website', min_value=0.0, value=37.0, step=0.1)
length_of_membership = st.number_input('Length of Membership', min_value=0.0, value=3.5, step=0.1)

# Predict button
if st.button('Predict Yearly Spend'):
    input_data = pd.DataFrame([[avg_session_length, time_on_app, time_on_website, length_of_membership]], 
                              columns=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'])
    prediction = lm.predict(input_data)
    st.write(f"Predicted Yearly Spend: ${prediction[0]:.2f}")
    </code></pre>
    <h2>Future Directions</h2>
    <ul>
        <li><strong>Model Enhancement</strong>: Explore advanced machine learning models (e.g., Decision Trees, Random Forest, or Gradient Boosting) to improve prediction accuracy.</li>
        <li><strong>Data Augmentation</strong>: Increase the dataset by including more features such as user demographics (age, gender), product preferences, or geographic location for more granular predictions.</li>
        <li><strong>User Interface</strong>: Enhance the interface by adding features like error handling, input validation, and visualization of prediction results (e.g., graphs or charts).</li>
        <li><strong>Model Deployment</strong>: Deploy the app on a cloud platform like <strong>Heroku</strong> or <strong>Streamlit Cloud</strong> for wider accessibility.</li>
        <li><strong>Real-time Data</strong>: Connect the app to live data or user activity tracking to provide real-time predictions for active users.</li>
    </ul>
    <h2>Acknowledgments</h2>
    <ul>
        <li><strong>Streamlit</strong>: For building the interactive web app.</li>
        <li><strong>Scikit-learn</strong>: For providing the machine learning algorithms and tools.</li>
        <li><strong>Pandas</strong>: For data manipulation and analysis.</li>
    </ul>
