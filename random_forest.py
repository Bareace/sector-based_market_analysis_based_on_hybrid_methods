import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime


def train_rf_model_from_excel(stock_code):
    stock_file_path = f'Data/clean_datas/merged_data_{stock_code}.xlsx'
    selected_columns = ['Date', 'Change%_stock', 'Change%_index', 'RSI', 'ATR', 'neg', 'neu', 'pos', 'compound',
                        'Relative_change']

    # Dosyayı okurken başlıkları direkt olarak kullan
    df = pd.read_excel(stock_file_path, usecols=selected_columns)

    # 'Date' sütununu datetime'a çevirip haftanın gününü çıkar
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_Week'] = df['Date'].dt.dayofweek  # Haftanın günü, Pazartesi=0, Pazar=6

    # Haftanın günü bilgisini kullanarak modeli eğit
    features_kap = df[['RSI', 'ATR', 'Change%_stock', 'Change%_index', 'Day_of_Week', 'neg', 'pos']].copy()
    features_kap_without_ATR = df[['RSI', 'Change%_stock', 'Change%_index', 'Day_of_Week', 'neg', 'pos']].copy()
    features = df[['RSI', 'ATR', 'Change%_stock', 'Change%_index', 'Day_of_Week']].copy()
    target = df['Relative_change'].copy()

    # Özellikleri ölçeklendir
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_kap[['RSI', 'ATR', 'Change%_stock', 'Change%_index']])
    features_kap[['RSI', 'ATR', 'Change%_stock', 'Change%_index']] = features_scaled

    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(features_kap, target, test_size=0.3, random_state=42)
    print(X_train.count() + X_test.count())
    # Modeli eğit
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Modelin performansını test et
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('Confusion Matrix:')
    print(cm)


# train_rf_model_from_excel('ASELS.IS_clean_max1')         # 0.46, clean: 0.63
# train_rf_model_from_excel('LOGO.IS_clean')    # 0.49, clean: 0.50
# train_rf_model_from_excel('TCELL.IS_clean_5')  # 0.45, clean: 0.47_3, 0.43_4, 0.54_5
train_rf_model_from_excel('clean')  # 0.54, ATR'siz: 0.62
