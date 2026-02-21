import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall

# --- DOSYA YOLU AYARLARI ---
mevcut_klasor = os.path.dirname(os.path.abspath(__file__))
csv_yolu = os.path.join(mevcut_klasor, "plant_health_data.csv")

if not os.path.exists(csv_yolu):
    print(f"HATA: '{csv_yolu}' bulunamadı!")
    exit()

print("⏳ Veri işleniyor...")
df = pd.read_csv(csv_yolu)

features = ['Soil_Moisture', 'Soil_Temperature', 'Soil_pH', 'Nitrogen_Level',
            'Phosphorus_Level', 'Potassium_Level', 'Ambient_Temperature',
            'Humidity', 'Light_Intensity', 'Chlorophyll_Content',
            'Electrochemical_Signal']

X = df[features].values

# Etiketleme
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Plant_Health_Status'])
y_categorical = to_categorical(y_encoded, num_classes=3)

# Scaler (BU ÇOK ÖNEMLİ - Kaydedip aynısını arayüzde kullanacağız)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim/Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Sınıf Ağırlıkları
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights = dict(enumerate(class_weights_array))

# Model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Eğitim Başlıyor...")
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=0, class_weight=class_weights)
print("✅ Eğitim Tamamlandı.")

# --- KAYDETME ---
print("💾 Dosyalar kaydediliyor...")
model.save(os.path.join(mevcut_klasor, "plant_health_tf_model.h5"))
joblib.dump(scaler, os.path.join(mevcut_klasor, "scaler.joblib"))
joblib.dump(label_encoder, os.path.join(mevcut_klasor, "label_encoder.joblib"))
print("✅ Tüm dosyalar başarıyla yenilendi!")

# --- TEK PENCEREDE TÜM GRAFİKLER ---
plt.figure(figsize=(15, 5))

# 1. Kayıp/Doğruluk
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Test Başarısı')
plt.title('Model Başarısı')
plt.xlabel('Epoch')
plt.legend()

# 2. Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.subplot(1, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Karmaşıklık Matrisi')
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')

# 3. Rapor (Metin olarak grafiğe ekleyelim)
plt.subplot(1, 3, 3)
plt.axis('off')
plt.text(0.1, 0.5, classification_report(y_true, y_pred, target_names=label_encoder.classes_), fontsize=10, family='monospace')
plt.title('Detaylı Rapor')

plt.tight_layout()
print("📊 Grafikler gösteriliyor. Devam etmek için pencereyi kapatın.")
plt.show()