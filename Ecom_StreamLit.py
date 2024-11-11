import pandas as pd #provides data exploratory and manipulation options/self explanatory with use
import matplotlib.pyplot as plt #would allow basic plotting of data set
import seaborn as sns #highlevel interface plots with easy linear regression plots
df = pd.read_csv('/ecommerce.csv') #loading the data into the colab notebook
df.head() #displaying the data set
df.info() #gives more information about the data set
df.describe() #give more statistical information about the data
#EDA
#Exploratory Data Analysis

sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df, alpha=0.5)
# alpha is used to provide greator opacity based on more occurance of points
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df, alpha=0.5)
# we would see a faint co-relation between both variable
sns.pairplot(df, kind='scatter', plot_kws={'alpha': 0.4})
#used to produce scatter plots between each variable of the dataset, letting co-relation intuition being better
#kind - controls type of plot, alpha - controls opacity relative to occurance
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df, scatter_kws={'alpha': 0.3})
#creates a linear regression plot with a regression line and confidence interval
#sklearn is a machine learning models library where we can simply import and use the models
from sklearn.model_selection import train_test_split
#dividing our dataset into a training set and a testing set
x = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
# fetching data as x and y list variables
#creating training and testing split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#test_size being 0.3 means 30% of the data is for testing while 70% is for training
#the random state controls the seed value to randomize the splitting of data everytime we run the code
from sklearn.linear_model import LinearRegression
#importing the linear regression model
lm=LinearRegression()
#creating a linear regression object
lm.fit(X_train, y_train)
#fitting the linear regression model to the training data
#finding the optimal parameters/coeff that would describe the importance of each feature in the polynomic linear regression model
#knowing these coeff would make the model rightfully use each coeff to describe the imp of each feature while predicting output
lm.coef_   
cdf=pd.DataFrame(lm.coef_, x.columns, columns=['Coeff'])
print(cdf) 
#predictions
predictions=lm.predict(X_test)
#the outputs that the model thinks would appear when we would use the input of the X_test
predictions      
#comparing the predicted values with the actual outputs
sns.scatterplot(x=predictions, y=y_test)
plt.xlabel('Predicted')
plt.ylabel('Actual')
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
#this metric calculates the average absolute difference between the predicted values (predictions) and the actual values (y_test).
print ("mean absolute error:", mean_absolute_error(y_test, predictions))
#This metric calculates the average of the squared differences between the predicted and actual values.
#By squaring the errors, MSE gives more weight to larger errors, which can be helpful if you want to penalize larger deviations.
print ("mean squared error:", mean_squared_error(y_test, predictions))
#RMSE is simply the square root of MSE, bringing the units back to the original scale of the target variable.
#RMSE is often used because it penalizes larger errors more heavily (like MSE) while remaining interpretable on the same scale as the data.
print ("root mean squared error:", math.sqrt(mean_squared_error(y_test, predictions)))
#residuals
#differences between the actual values and the predicted values
residuals=y_test-predictions
#Positive Residuals: Indicate that the model underestimated the actual value.
#Negative Residuals: Indicate that the model overestimated the actual value.
#Zero Residual: Means the model’s prediction was exactly correct for that instance.
print(residuals)
sns.distplot((y_test-predictions), bins=50)
import pylab
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=pylab)
pylab.show()