import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from datetime import datetime

def create_model(lstm_units=50, dropout_rate=0.2, num_classes=3):
    model = Sequential([
        LSTM(lstm_units, input_shape=(1, 5), return_sequences=True),  # Girdi şeklini 5 özellik olarak güncelledim
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_lstm_model(stock_code):
    stock_file_path = f'Data/merged_data_{stock_code}.xlsx'
    selected_columns = ['Date', 'Change%_stock', 'Change%_index', 'RSI', 'ATR', 'Relative_change']

    df = pd.read_excel(stock_file_path, usecols=selected_columns)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_Week'] = df['Date'].dt.dayofweek

    features = df[['RSI', 'ATR', 'Change%_stock', 'Change%_index', 'Day_of_Week']].copy()
    target = df['Relative_change'].copy()

    # Encode the target variable
    encoder = LabelEncoder()
    target_encoded = encoder.fit_transform(target)
    num_classes = len(np.unique(target_encoded))  # Dinamik olarak sınıf sayısını belirle

    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Convert target to categorical
    target_encoded = to_categorical(target_encoded)

    # Reshape input to be [samples, time steps, features]
    features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_encoded, test_size=0.3, random_state=42)

    # Create and train the model
    model = create_model(lstm_units=50, dropout_rate=0.2, num_classes=num_classes)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'LSTM Model Accuracy: {accuracy:.2f}')

#train_lstm_model('ASELS.IS_clean')
#train_model_from_excel('LOGO.IS_clean')
train_lstm_model('TCELL.IS')