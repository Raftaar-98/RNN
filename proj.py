# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


Data = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/RNN/main/AMAZON_daily.csv")
# Load the dataset
class LSTM(object):
    def preprocess(self):
        # Convert 'Date' column to datetime
        Data['Date'] = pd.to_datetime(Data['Date'])

        # Filter data for the last 5 years
        start_date = '2018-11-01'
        end_date = '2023-10-31'
        Data = Data[(Data['Date'] >= start_date) & (Data['Date'] <= end_date)]

        # Select relevant columns for modeling
        selected_columns = ['Date', 'Close']  # Add other columns as needed
        Data = Data[selected_columns]

        # Set 'Date' column as the index
        Data.set_index('Date', inplace=True)

        # Feature scaling (normalize the data)
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(Data), columns=Data.columns, index=Data.index)
            # Define the number of previous time steps for prediction
        time_steps = 30  # Adjust as needed
        X, y = self.create_sequences(df_scaled, time_steps)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train,y_test

    # Function to create time series sequences
    def create_sequences(self,data, time_steps=1):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data.iloc[i:(i + time_steps)].values)
            y.append(data.iloc[i + time_steps].values)
        return np.array(X), np.array(y)


if __name__=='__main__':
    model = LSTM()
    X,y,T_X,T_y = model.preprocess()
    train_X = X # split training data and testing data
    train_y = y
    test_X = T_X
    test_y = T_y
    print(train_X)
    print(train_y)
    print(test_X)
    print(test_y)
# Split the data into training and test sets

'''''
# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Invert scaling for predictions and actual values
y_pred_actual = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Evaluate the model
mse = mean_squared_error(y_test_actual, y_pred_actual)
print("Mean Squared Error:", mse)

# Plot the predictions and actual values
plt.figure(figsize=(12, 6))
plt.plot(Data.index[-len(y_test_actual):], y_test_actual, label='Actual Stock Price', color='blue')
plt.plot(Data.index[-len(y_test_actual):], y_pred_actual, label='Predicted Stock Price', color='red')
plt.title('Stock Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
'''''