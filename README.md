<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-commerce Customer Analysis with Linear Regression</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2 {
            color: #0073e6;
            text-align: center;
        }
        h3, h4 {
            color: #005bb5;
        }
        p {
            margin: 10px 0;
        }
        ul, ol {
            margin: 10px 0 20px 20px;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 5px;
            color: #d63384;
            font-weight: bold;
        }
        pre {
            background: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
            border-left: 4px solid #0073e6;
        }
        .section {
            padding: 10px;
            margin: 20px 0;
            border-radius: 8px;
            background-color: #eef7ff;
        }
    </style>
</head>
<body>
    <h1>Project Name: E-commerce Customer Analysis with Linear Regression</h1>
    <h2>README</h2>
    <div class="section">
        <h3>Project Purpose</h3>
        <p>This project analyzes customer behavior in an e-commerce setting to predict yearly spending using linear regression. It involves data exploration, feature engineering, model training, and insights on customer spending patterns.</p>
    </div>
    <div class="section">
        <h3>Data Requirements</h3>
        <p>Ensure that the dataset <code>ecommerce.csv</code> is in the same directory as the code file. The dataset can be downloaded from the repository or from Kaggle if not already included.</p>
    </div>
    <div class="section">
        <h3>Procedure Overview</h3>
        <ol>
            <li><strong>Data Loading & Exploration:</strong> Load, inspect, and visualize data for initial insights.</li>
            <li><strong>Feature Selection & Model Setup:</strong> Select features and train a linear regression model.</li>
            <li><strong>Model Evaluation:</strong> Use error metrics and visualization for performance analysis.</li>
            <li><strong>Insights:</strong> Analyze feature impact and residual distribution.</li>
        </ol>
    </div>
    <div class="section">
        <h3>Step-by-Step Guide</h3>
        <h4>Step 1: Import Libraries</h4>
        <ul>
            <li><strong>Pandas</strong> - data handling</li>
            <li><strong>Matplotlib & Seaborn</strong> - visualization</li>
            <li><strong>Scikit-learn</strong> - machine learning</li>
            <li><strong>SciPy</strong> - statistical analysis</li>
        </ul>
        <pre><code>import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy.stats as stats</code></pre>
        <h4>Step 2: Data Loading & Initial Exploration</h4>
        <p>Load the data and check the structure:</p>
        <pre><code>df = pd.read_csv('ecommerce.csv')
df.head()</code></pre>
        <h4>Step 3: Exploratory Data Analysis (EDA)</h4>
        <p>Visualize relationships with joint plots and pair plots:</p>
        <pre><code>sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df, alpha=0.5)
sns.pairplot(df, plot_kws={'alpha': 0.4})</code></pre>
        <h4>Step 4: Data Splitting & Model Training</h4>
        <p>Split data and train the model:</p>
        <pre><code>x = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
lm = LinearRegression()
lm.fit(X_train, y_train)</code></pre>
        <h4>Step 5: Model Interpretation</h4>
        <p>View feature impact with model coefficients:</p>
        <pre><code>cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coeff'])</code></pre>
        <h4>Step 6: Predictions and Visualization</h4>
        <p>Plot predicted values against actual values:</p>
        <pre><code>predictions = lm.predict(X_test)
sns.scatterplot(x=predictions, y=y_test)</code></pre>
        <h4>Step 7: Performance Metrics</h4>
        <p>Evaluate using MAE, MSE, and RMSE:</p>
        <pre><code>from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", math.sqrt(mean_squared_error(y_test, predictions)))</code></pre>
        <h4>Step 8: Residual Analysis</h4>
        <p>Verify residuals for model fit assessment:</p>
        <pre><code>residuals = y_test - predictions
sns.histplot(residuals, bins=30)</code></pre>
    </div>
    <div class="section">
        <h3>Results</h3>
        <p>The model shows strong predictive performance with meaningful features. Residuals follow a near-normal distribution, supporting model fit.</p>
    </div>
    <div class="section">
        <h3>Applications</h3>
        <ul>
            <li><strong>Marketing:</strong> Predict spending for targeted campaigns.</li>
            <li><strong>Customer Retention:</strong> Identify high-value customer characteristics.</li>
            <li><strong>Business Decisions:</strong> Data-driven insights for strategic planning.</li>
        </ul>
    </div>
    <div class="section">
        <h3>Instructions to Run</h3>
        <ol>
            <li>Ensure Python and libraries are installed.</li>
            <li>Download <code>ecommerce.csv</code> and place it in the project folder.</li>
            <li>Run each section in a Jupyter Notebook or compatible IDE to analyze results.</li>
        </ol>
    </div>
</body>
</html>
