import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# 1. DOSYAYI BUL
mevcut_klasor = os.path.dirname(os.path.abspath(__file__))
csv_yolu = os.path.join(mevcut_klasor, "plant_health_data.csv")

if not os.path.exists(csv_yolu):
    print("❌ HATA: CSV dosyası bulunamadı!")
    exit()

print(f"📂 Dosya okunuyor: {csv_yolu}")
df = pd.read_csv(csv_yolu)

# 2. VERİ ARALIKLARINI GÖSTER (EN ÖNEMLİ KISIM)
features = ['Soil_Moisture', 'Soil_Temperature', 'Soil_pH', 'Nitrogen_Level',
            'Phosphorus_Level', 'Potassium_Level', 'Ambient_Temperature',
            'Humidity', 'Light_Intensity', 'Chlorophyll_Content',
            'Electrochemical_Signal']

print("\n📊 VERİ SETİNDEKİ GERÇEK ARALIKLAR:")
print("-" * 60)
print(f"{'Özellik Adı':<25} | {'Min Değer':<10} | {'Max Değer':<10} | {'Ortalama':<10}")
print("-" * 60)
for col in features:
    print(f"{col:<25} | {df[col].min():<10.2f} | {df[col].max():<10.2f} | {df[col].mean():<10.2f}")
print("-" * 60)

# 3. HIZLI EĞİTİM (Arayüzsüz)
print("\n⚙️  Model hızlıca eğitiliyor (Test için)...")
X = df[features].values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Plant_Health_Status'])
y_categorical = to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Scaler burada eğitildi

model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y_categorical, epochs=20, verbose=0) 

# 4. MANUEL TEST (Senin girdiğin değerler)
# Sağlıklı bitki senaryosu değerlerin:
senaryo_degerleri = [75, 22, 6.5, 80, 60, 70, 24, 65, 3000, 600, 200]

# Bu değerleri modele sormadan önce SCALER ile dönüştürüyoruz
test_verisi = np.array([senaryo_degerleri])
test_scaled = scaler.transform(test_verisi)

tahmin = model.predict(test_scaled, verbose=0)[0]
siniflar = label_encoder.classes_

print("\n🧪 TEST SONUCU (Girdiğin [75, 22, 6.5...] değerleri için):")
print("-" * 40)
for i, sinif in enumerate(siniflar):
    print(f"{sinif}: %{tahmin[i]*100:.2f}")
print("-" * 40)