import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Student_Performance.csv')

print (data.head())
print (data.info())
print (data.isnull().sum())
print (data.shape)
print (data.duplicated().sum())

data.drop_duplicates(inplace=True)

print(data.describe().T)

# Check correlation using a heatmap
sns.heatmap(data=data.select_dtypes(exclude="object").corr(), annot=True, cmap="vlag")
plt.show()

# Prepare for training, testing split - All features in x, the predicted value in y
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
# x will now contain Hours studied, Previous scores, Extracurricluar activities, Sleep Hours, and Sample Question Papers Practiced (Features)
# y will contain the outcome (Performance Index)

print(x)
print(y)

# Do one-hot encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label = LabelEncoder()
x[:,2]= label.fit_transform(x[:,2])
print(x)

# Split data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

print(y_train.shape)
print(y_test.shape)

# Perform linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

# Print results
pred = model.predict(x_test)
print('Train score :{} \n Test score {}'.format(model.score(x_train,y_train),model.score(x_test,y_test)))

print(pred)

# Check accuracy
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
print('MSE : {} \n Mean absolute error : {} \n r2_score :{}'.format( mean_squared_error(y_test,pred), mean_absolute_error(y_test,pred), r2_score(y_test,pred)))

# Create a new DF containing actual Performance Index value and predicted Performance Index value for the test set

df_compare = pd.DataFrame()
df_compare['actual'] = y_test
df_compare['predicted'] = pred

print(df_compare)

# Create regression plot of the actual versus predicted
correlation_coefficient = df_compare['actual'].corr(df_compare['predicted'])

print(f'Correlation Coefficient: {correlation_coefficient}')

# Plot a scatter plot to visualize the data
plt.scatter(df_compare['actual'], df_compare['predicted'])
plt.xlabel('Actual Performance Index Value')
plt.ylabel('Predicted Performance Index Value')
plt.title(f'Scatter Plot (Correlation: {correlation_coefficient:.2f})')
plt.show()