import tkinter as tk
from tkinter import ttk
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train_rf_model_from_excel(stock_code, exclude_atr=False):
    stock_file_path = f'Data/clean_datas/merged_data_{stock_code}.xlsx'
    selected_columns = ['Date', 'Change%_stock', 'Change%_index', 'RSI', 'ATR', 'neg', 'neu', 'pos', 'compound',
                        'Relative_change']

    df = pd.read_excel(stock_file_path, usecols=selected_columns)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_Week'] = df['Date'].dt.dayofweek

    if exclude_atr:
        features_kap = df[['RSI', 'Change%_stock', 'Change%_index', 'Day_of_Week', 'neg', 'pos']].copy()
        features = df[['RSI', 'Change%_stock', 'Change%_index', 'Day_of_Week']].copy()
    else:
        features_kap = df[['RSI', 'ATR', 'Change%_stock', 'Change%_index', 'Day_of_Week', 'neg', 'pos']].copy()
        features = df[['RSI', 'ATR', 'Change%_stock', 'Change%_index', 'Day_of_Week']].copy()
    target = df['Relative_change'].copy()

    scaler = StandardScaler()
    features_kap_scaled = scaler.fit_transform(features_kap)
    features_scaled = scaler.fit_transform(features)

    X_train_kap, X_test_kap, y_train_kap, y_test_kap = train_test_split(features_kap_scaled, target, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

    model_kap = RandomForestClassifier(n_estimators=100, random_state=42)
    model_kap.fit(X_train_kap, y_train_kap)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_kap = model_kap.predict(X_test_kap)
    accuracy_kap = accuracy_score(y_test_kap, y_pred_kap)
    precision_kap = precision_score(y_test_kap, y_pred_kap, average='macro')
    recall_kap = recall_score(y_test_kap, y_pred_kap, average='macro')
    f1_kap = f1_score(y_test_kap, y_pred_kap, average='macro')
    cm_kap = confusion_matrix(y_test_kap, y_pred_kap)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'accuracy_kap': accuracy_kap,
        'precision_kap': precision_kap,
        'recall_kap': recall_kap,
        'f1_kap': f1_kap,
        'cm_kap': cm_kap,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm
    }

    return results


def create_model(lstm_units=50, dropout_rate=0.2, num_classes=3, input_shape=(1, 6)):
    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_lstm_model(stock_code, exclude_atr=False):
    stock_file_path = f'Data/clean_datas/merged_data_{stock_code}.xlsx'
    selected_columns = ['Date', 'Change%_stock', 'Change%_index', 'RSI', 'ATR', 'neg', 'neu', 'pos', 'Relative_change']

    df = pd.read_excel(stock_file_path, usecols=selected_columns)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_Week'] = df['Date'].dt.dayofweek

    if exclude_atr:
        features_kap = df[['RSI', 'Change%_stock', 'Change%_index', 'Day_of_Week', 'neg', 'pos']].copy()
        features = df[['RSI', 'Change%_stock', 'Change%_index', 'Day_of_Week']].copy()
    else:
        features_kap = df[['RSI', 'ATR', 'Change%_stock', 'Change%_index', 'Day_of_Week', 'neg', 'pos']].copy()
        features = df[['RSI', 'ATR', 'Change%_stock', 'Change%_index', 'Day_of_Week']].copy()
    target = df['Relative_change'].copy()

    encoder = LabelEncoder()
    target_encoded = encoder.fit_transform(target)
    num_classes = len(np.unique(target_encoded))
    target_encoded = to_categorical(target_encoded)

    scaler = StandardScaler()
    features_kap_scaled = scaler.fit_transform(features_kap)
    features_scaled = scaler.fit_transform(features)

    features_kap_scaled = np.reshape(features_kap_scaled, (features_kap_scaled.shape[0], 1, features_kap_scaled.shape[1]))
    features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

    X_train_kap, X_test_kap, y_train_kap, y_test_kap = train_test_split(features_kap_scaled, target_encoded, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_encoded, test_size=0.3, random_state=42)

    input_shape_kap = (1, features_kap_scaled.shape[2])
    input_shape = (1, features_scaled.shape[2])

    model_kap = create_model(lstm_units=50, dropout_rate=0.2, num_classes=num_classes, input_shape=input_shape_kap)
    model = create_model(lstm_units=50, dropout_rate=0.2, num_classes=num_classes, input_shape=input_shape)

    model_kap.fit(X_train_kap, y_train_kap, epochs=50, batch_size=32, validation_data=(X_test_kap, y_test_kap))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    y_pred_kap = np.argmax(model_kap.predict(X_test_kap), axis=1)
    y_test_kap_labels = np.argmax(y_test_kap, axis=1)
    accuracy_kap = accuracy_score(y_test_kap_labels, y_pred_kap)
    precision_kap = precision_score(y_test_kap_labels, y_pred_kap, average='macro')
    recall_kap = recall_score(y_test_kap_labels, y_pred_kap, average='macro')
    f1_kap = f1_score(y_test_kap_labels, y_pred_kap, average='macro')
    cm_kap = confusion_matrix(y_test_kap_labels, y_pred_kap)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, y_pred)
    precision = precision_score(y_test_labels, y_pred, average='macro')
    recall = recall_score(y_test_labels, y_pred, average='macro')
    f1 = f1_score(y_test_labels, y_pred, average='macro')
    cm = confusion_matrix(y_test_labels, y_pred)

    results = {
        'accuracy_kap': accuracy_kap,
        'precision_kap': precision_kap,
        'recall_kap': recall_kap,
        'f1_kap': f1_kap,
        'cm_kap': cm_kap,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm
    }

    return results


def train_gradient_boosting_model_from_excel(stock_code, exclude_atr=False):
    stock_file_path = f'Data/clean_datas/merged_data_{stock_code}.xlsx'
    selected_columns = ['Date', 'Change%_stock', 'Change%_index', 'RSI', 'ATR', 'neg', 'neu', 'pos', 'compound',
                        'Relative_change']

    df = pd.read_excel(stock_file_path, usecols=selected_columns)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_Week'] = df['Date'].dt.dayofweek

    if exclude_atr:
        features_kap = df[['RSI', 'Change%_stock', 'Change%_index', 'Day_of_Week', 'neg', 'pos']].copy()
        features = df[['RSI', 'Change%_stock', 'Change%_index', 'Day_of_Week']].copy()
    else:
        features_kap = df[['RSI', 'ATR', 'Change%_stock', 'Change%_index', 'Day_of_Week', 'neg', 'pos']].copy()
        features = df[['RSI', 'ATR', 'Change%_stock', 'Change%_index', 'Day_of_Week']].copy()
    target = df['Relative_change'].copy()

    scaler = StandardScaler()
    features_kap_scaled = scaler.fit_transform(features_kap)
    features_scaled = scaler.fit_transform(features)

    X_train_kap, X_test_kap, y_train_kap, y_test_kap = train_test_split(features_kap_scaled, target, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

    model_kap = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    model_kap.fit(X_train_kap, y_train_kap)
    model.fit(X_train, y_train)

    y_pred_kap = model_kap.predict(X_test_kap)
    accuracy_kap = accuracy_score(y_test_kap, y_pred_kap)
    precision_kap = precision_score(y_test_kap, y_pred_kap, average='macro')
    recall_kap = recall_score(y_test_kap, y_pred_kap, average='macro')
    f1_kap = f1_score(y_test_kap, y_pred_kap, average='macro')
    cm_kap = confusion_matrix(y_test_kap, y_pred_kap)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'accuracy_kap': accuracy_kap,
        'precision_kap': precision_kap,
        'recall_kap': recall_kap,
        'f1_kap': f1_kap,
        'cm_kap': cm_kap,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm
    }

    return results


# Ana pencereyi oluştur
root = tk.Tk()
root.title("Hisse Senedi Tahmin Aracı")

# Dropdown menü seçenekleri
hisse_adlari = ["Aselsan", "Turkcell", "Logo", "Mia Bilişim", "Tüm Şirketler"]
model_adlari = ["Random Forest", "Gradient Boosting Machines", "LSTM"]
gpt4_desteği = ["Var", "Yok"]
haber_etkisi = ["Yeni haber gelene kadar", "3 güne kadar"]

# İlk dropdown (Hisse Adı)
ttk.Label(root, text="Hisse Adı:").grid(column=0, row=0, padx=10, pady=10)
hisse_adi_var = tk.StringVar()
hisse_adi_dropdown = ttk.Combobox(root, textvariable=hisse_adi_var, values=hisse_adlari)
hisse_adi_dropdown.grid(column=1, row=0, padx=10, pady=10)
hisse_adi_dropdown.current(0)

# İkinci dropdown (Model Adı)
ttk.Label(root, text="Model Adı:").grid(column=0, row=1, padx=10, pady=10)
model_adi_var = tk.StringVar()
model_adi_dropdown = ttk.Combobox(root, textvariable=model_adi_var, values=model_adlari)
model_adi_dropdown.grid(column=1, row=1, padx=10, pady=10)
model_adi_dropdown.current(0)

# # Üçüncü dropdown (GPT-4 Desteği)
# ttk.Label(root, text="GPT-4 Desteği:").grid(column=0, row=2, padx=10, pady=10)
# gpt4_desteği_var = tk.StringVar()
# gpt4_desteği_dropdown = ttk.Combobox(root, textvariable=gpt4_desteği_var, values=gpt4_desteği)
# gpt4_desteği_dropdown.grid(column=1, row=2, padx=10, pady=10)
# gpt4_desteği_dropdown.current(0)

# Dördüncü dropdown (Haber Etkisi)
ttk.Label(root, text="Haber Etkisi:").grid(column=0, row=3, padx=10, pady=10)
haber_etkisi_var = tk.StringVar()
haber_etkisi_dropdown = ttk.Combobox(root, textvariable=haber_etkisi_var, values=haber_etkisi)
haber_etkisi_dropdown.grid(column=1, row=3, padx=10, pady=10)
haber_etkisi_dropdown.current(0)

# Sonuçları göstermek için metin alanı
results_text = tk.Text(root, height=15, width=50)
results_text.grid(column=0, row=5, columnspan=2, padx=10, pady=10)


# Buton tıklama işlemi için fonksiyon
def get_results():
    results_text.delete(1.0, tk.END)
    results_text.insert(tk.END, "Getting Results...\n")
    root.update_idletasks()  # Update the text widget to show the message immediately

    exclude_atr = False
    hisse_kodu = ""
    hisse_adi = hisse_adi_var.get()
    model_adi = model_adi_var.get()
    # gpt4_destegi = gpt4_desteği_var.get()
    haber_etkisi = haber_etkisi_var.get()

    if hisse_adi == "Aselsan":
        hisse_kodu += "ASELS.IS"
    elif hisse_adi == "Turkcell":
        hisse_kodu += "TCELL.IS"
    elif hisse_adi == "Logo":
        hisse_kodu += "LOGO.IS"
    elif hisse_adi == "Mia Bilişim":
        hisse_kodu += "MIATK.IS"
    elif hisse_adi == "Tüm Şirketler":
        hisse_kodu += "merged"
        exclude_atr = True

    if haber_etkisi == "3 güne kadar":
        hisse_kodu += "_clean_max3"

    # Seçilen modele göre ilgili fonksiyonu çağır
    if model_adi == "Random Forest":
        results = train_rf_model_from_excel(hisse_kodu, exclude_atr=exclude_atr)
    elif model_adi == "Gradient Boosting Machines":
        results = train_gradient_boosting_model_from_excel(hisse_kodu, exclude_atr=exclude_atr)
    elif model_adi == "LSTM":
        results = train_lstm_model(hisse_kodu, exclude_atr=exclude_atr)

    # Sonuçları metin alanında göster
    results_text.delete(1.0, tk.END)
    results_text.insert(tk.END, "Results with KAP Data:\n")
    results_text.insert(tk.END, f"Accuracy: {results['accuracy_kap']:.2f}\n")
    if isinstance(results.get('precision_kap'), (float, int)):
        results_text.insert(tk.END, f"Precision: {results['precision_kap']:.2f}\n")
    else:
        results_text.insert(tk.END, "Precision: N/A\n")
    if isinstance(results.get('recall_kap'), (float, int)):
        results_text.insert(tk.END, f"Recall: {results['recall_kap']:.2f}\n")
    else:
        results_text.insert(tk.END, "Recall: N/A\n")
    if isinstance(results.get('f1_kap'), (float, int)):
        results_text.insert(tk.END, f"F1 Score: {results['f1_kap']:.2f}\n")
    else:
        results_text.insert(tk.END, "F1 Score: N/A\n")

    results_text.insert(tk.END, "\nResults without KAP Data:\n")
    results_text.insert(tk.END, f"Accuracy: {results['accuracy']:.2f}\n")
    if isinstance(results.get('precision'), (float, int)):
        results_text.insert(tk.END, f"Precision: {results['precision']:.2f}\n")
    else:
        results_text.insert(tk.END, "Precision: N/A\n")
    if isinstance(results.get('recall'), (float, int)):
        results_text.insert(tk.END, f"Recall: {results['recall']:.2f}\n")
    else:
        results_text.insert(tk.END, "Recall: N/A\n")
    if isinstance(results.get('f1'), (float, int)):
        results_text.insert(tk.END, f"F1 Score: {results['f1']:.2f}\n")
    else:
        results_text.insert(tk.END, "F1 Score: N/A\n")


# Sonuçları Getir butonu
get_results_button = ttk.Button(root, text="Sonuçları Getir", command=get_results)
get_results_button.grid(column=0, row=4, columnspan=2, padx=10, pady=10)

# Ana döngüyü başlat
root.mainloop()