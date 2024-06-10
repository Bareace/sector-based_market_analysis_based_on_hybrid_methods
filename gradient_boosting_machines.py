#Gradient Boosting Machines (GBM)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train_gradient_boosting_model_from_excel(stock_code):
    stock_file_path = f'Data/merged_data_{stock_code}.xlsx'
    # stock_file_path = f'/content/drive/My Drive/Data/merged_data_{stock_code}.xlsx'
    # stock_file_path = f'/content/drive/My Drive/Data/{stock_code}.xlsx'
    selected_columns = ['Date','Change%_stock','Change%_index', 'RSI', 'ATR', 'neg', 'neu', 'pos', 'compound', 'Relative_change']

    # Dosyayı okurken başlıkları direkt olarak kullan
    df = pd.read_excel(stock_file_path, usecols=selected_columns)

    # 'Date' sütununu datetime'a çevirip haftanın gününü çıkar
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_Week'] = df['Date'].dt.dayofweek  # Haftanın günü, Pazartesi=0, Pazar=6

    # Haftanın günü bilgisini kullanarak modeli eğit
    features_kap = df[['RSI', 'ATR','Change%_stock','Change%_index', 'Day_of_Week', 'neg', 'pos']].copy()
    features_kap_without_ATR = df[['RSI','Change%_stock','Change%_index', 'Day_of_Week', 'neg', 'pos']].copy()
    features = df[['RSI', 'ATR','Change%_stock','Change%_index', 'Day_of_Week']].copy()
    target = df['Relative_change'].copy()

    # Özellikleri ölçeklendir
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_kap[['RSI', 'Change%_stock','Change%_index']])
    features_kap[['RSI', 'Change%_stock','Change%_index']] = features_scaled

    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(features_kap, target, test_size=0.3, random_state=42)

    # Modeli eğit
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
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

train_gradient_boosting_model_from_excel('ASELS.IS')  