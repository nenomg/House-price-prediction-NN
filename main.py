# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 22:08:34 2023

@author: NENO
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

#Plot the correlation matrix and filter in the dataset the columns that are 
#correlated > 0.4
def plotCorrelation(data):
    
    # Calculate the correlation matrix
    correlation_matrix = data.corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 12))
    
    # Create a heatmap of the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    
    # Customize the plot (optional)
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Filter rows with correlation >= 0.4 with the "price" column
    correlated_rows = correlation_matrix.index[correlation_matrix['price'] >= -1].tolist()
    
    # Remove rows that are not in the correlated list
    data = data[correlated_rows]
    
    return data

def plotPricesScatter(data, columns):
    prices = data['price'][0:columns]
    plt.scatter(range(len(prices)), prices)
    
    # Adding labels and title
    plt.xlabel('Data Points')
    plt.ylabel('Price')
    plt.title('Scatter Plot of Price')
    # Show the plot
    plt.show()

# Load the dataset
data = pd.read_csv("kc_house_data.csv")

# Remove the "date" column
data = data.drop(columns=['date'])

# Remove rows with correlation < 0.4
print("\n\nCorrelation matrix\n")

data = plotCorrelation(data)

print("\n\nFiltered Correlation Matrix\n")
data = plotCorrelation(data)

print("\n\nPrices\n")
plotPricesScatter(data, 100)




y = data["price"].values
X = data.drop(columns=['price']).values

y_test = y[0:50]
X_test = X[0:50]

y_train = y[50:len(y)]
X_train = X[50:len(X)]


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#Create the model
model = Sequential()

# Input layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
# Hidden layers
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
# Output layer for regression (linear activation)
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Print a summary of the model architecture
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, verbose=1)

# Plot the training loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Make predictions and plot them
predictions = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, label='Actual Prices')
plt.scatter(range(len(predictions)), predictions, label='Predicted Prices', marker='x')
plt.xlabel('Data Points')
plt.ylabel('Price')
plt.legend()
plt.title('Scatter Plot of Actual and Predicted Price')

plt.show()

