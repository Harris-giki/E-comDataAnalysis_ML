<h1><strong>Project Name:</strong> E-commerce Customer Analysis with Linear Regression</h1>

<h2>README</h2>

<h3><strong>Project Purpose</strong></h3>
<p>The goal of this project is to analyze customer behavior within an e-commerce setting and predict yearly customer spending based on various features using a linear regression model. This project focuses on exploratory data analysis (EDA) to uncover relationships between features and target variables, followed by a machine learning model to make predictive insights.</p>

<h3><strong>Data Requirements</strong></h3>
<p>The project requires the dataset to be named <code>ecommerce.csv</code> and located in the same directory as the code file. The dataset can be obtained from the project repository or from Kaggle. Adjustments to the file path should be made if the data is saved in a different location.</p>

<h3><strong>Procedure Overview</strong></h3>
<ol>
    <li><strong>Data Loading and Exploration:</strong> Load the dataset, examine the structure, and perform initial statistical analyses. Visualize key relationships between features and target variables to gain insights.</li>
    <li><strong>Feature Engineering and Model Selection:</strong> Select relevant features based on correlation analysis. Apply a linear regression model using scikit-learn to predict the target variable.</li>
    <li><strong>Model Evaluation:</strong> Assess model performance using metrics like Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error. Visualize predictions and residuals to analyze the model's performance.</li>
    <li><strong>Interpretation and Insights:</strong> Interpret the model coefficients to understand feature importance. Assess residual distribution to ensure the model’s assumptions hold.</li>
</ol>

<h3><strong>Detailed Project Steps</strong></h3>

<h4>Step 1: Importing Essential Libraries</h4>
<p>Each library used in this project serves a distinct purpose:</p>

<ul>
    <li><strong>Pandas:</strong> Used for data manipulation and handling. It provides structures like DataFrames, which allow easy exploration and analysis.</li>
    <pre><code>import pandas as pd</code></pre>
    
    <li><strong>Matplotlib</strong> and <strong>Seaborn:</strong> Used for data visualization. Matplotlib provides foundational plotting capabilities, while Seaborn offers a high-level interface for easier and aesthetically pleasing visuals, especially for statistical plots.</li>
    <pre><code>import matplotlib.pyplot as plt
import seaborn as sns</code></pre>
    
    <li><strong>scikit-learn:</strong> A comprehensive machine learning library used to split data, train models, and evaluate their performance. In this project, <code>LinearRegression</code> is used for building the prediction model.</li>
    <pre><code>from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression</code></pre>
    
    <li><strong>SciPy:</strong> Enables probability plotting, helpful for validating residual distribution.</li>
    <pre><code>import scipy.stats as stats</code></pre>
</ul>

<h4>Step 2: Data Loading and Initial Exploration</h4>
<p>Load the dataset using Pandas and inspect the first few rows to understand the structure. This step includes <code>df.info()</code> for summary information and <code>df.describe()</code> for statistical data.</p>
<pre><code>df = pd.read_csv('/ecommerce.csv')
df.head()
df.info()
df.describe()</code></pre>

<h4>Step 3: Exploratory Data Analysis (EDA)</h4>
<p>To visualize relationships between variables:</p>

<ul>
    <li><strong>Joint Plots:</strong> Used to visualize relationships between pairs, focusing on time spent on the website/app and yearly spending.</li>
    <pre><code>sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df, alpha=0.5)
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df, alpha=0.5)</code></pre>
    
    <li><strong>Pair Plot:</strong> Helps visualize all possible relationships between variables in the dataset.</li>
    <pre><code>sns.pairplot(df, kind='scatter', plot_kws={'alpha': 0.4})</code></pre>
    
    <li><strong>Linear Model Plot:</strong> Specifically targets the 'Length of Membership' and 'Yearly Amount Spent' features to analyze linear relationships.</li>
    <pre><code>sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df, scatter_kws={'alpha': 0.3})</code></pre>
</ul>

<h4>Step 4: Data Splitting and Model Training</h4>
<p>Split data into training and test sets with <code>train_test_split()</code>, allocating 70% for training and 30% for testing. Then, fit a Linear Regression model on the training data.</p>
<pre><code>x = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lm = LinearRegression()
lm.fit(X_train, y_train)</code></pre>

<h4>Step 5: Model Interpretation</h4>
<p>View the model coefficients to understand the impact of each feature.</p>
<pre><code>cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coeff'])
print(cdf)</code></pre>

<h4>Step 6: Prediction and Visualization</h4>
<p>Using the test data, generate predictions and plot them against actual values to evaluate performance.</p>
<pre><code>predictions = lm.predict(X_test)
sns.scatterplot(x=predictions, y=y_test)
plt.xlabel('Predicted')
plt.ylabel('Actual')</code></pre>

<h4>Step 7: Performance Metrics</h4>
<p>Evaluate model accuracy using three error metrics:</p>

<ul>
    <li><strong>Mean Absolute Error</strong> (MAE)</li>
    <li><strong>Mean Squared Error</strong> (MSE)</li>
    <li><strong>Root Mean Squared Error</strong> (RMSE)</li>
</ul>

<pre><code>from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("Root Mean Squared Error:", math.sqrt(mean_squared_error(y_test, predictions)))</code></pre>

<h4>Step 8: Residual Analysis</h4>
<p>Calculate and plot residuals to confirm a normal distribution, indicating a well-fitted model.</p>
<pre><code>residuals = y_test - predictions
sns.distplot((y_test - predictions), bins=50)
stats.probplot(residuals, dist="norm", plot=plt)
plt.show()</code></pre>

<h3><strong>Results</strong></h3>
<p>The model demonstrates a positive correlation between predicted and actual values of yearly spending, suggesting that the selected features have a strong predictive relationship. The error metrics are within acceptable limits, and residuals exhibit a near-normal distribution, further validating model effectiveness.</p>

<h3><strong>Relative Uses and Applications</strong></h3>
<ul>
    <li><strong>E-commerce Optimization:</strong> Predict customer spending to target marketing strategies.</li>
    <li><strong>Customer Retention:</strong> Identify factors that influence higher spending.</li>
    <li><strong>Business Strategy:</strong> Help data-driven decision-making based on customer behavior.</li>
</ul>

<h3><strong>How to Use</strong></h3>
<ol>
    <li><strong>Prerequisites:</strong> Ensure Python and necessary libraries (<code>pandas</code>, <code>matplotlib</code>, <code>seaborn</code>, <code>scikit-learn</code>, <code>scipy</code>) are installed.</li>
    <li><strong>Data File:</strong> Place <code>ecommerce.csv</code> in the same directory as this code file or adjust the path in <code>pd.read_csv()</code>.</li>
    <li><strong>Run the Notebook:</strong> Execute the code cells sequentially in Jupyter or any compatible IDE.</li>
    <li><strong>Analyze Output:</strong> Use generated plots and metrics to interpret the model's performance and insights.</li>
</ol>
