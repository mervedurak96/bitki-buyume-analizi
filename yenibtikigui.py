import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# --- DOSYALARI GÜVENLİ YÜKLEME ---
mevcut_klasor = os.path.dirname(os.path.abspath(__file__))

def dosya_yukle(dosya_adi, yukleyici_func):
    yol = os.path.join(mevcut_klasor, dosya_adi)
    if not os.path.exists(yol):
        messagebox.showerror("Eksik Dosya", f"'{dosya_adi}' bulunamadı!\nLütfen önce eğitim kodunu çalıştırın.")
        exit()
    return yukleyici_func(yol)

model = dosya_yukle("plant_health_tf_model.h5", load_model)
scaler = dosya_yukle("scaler.joblib", joblib.load)
label_encoder = dosya_yukle("label_encoder.joblib", joblib.load)

# Sabitler
ozellikler = ['Toprak Nem (%)', 'Toprak Sıcaklığı (°C)', 'Toprak pH',
              'Azot Seviyesi (mg/kg)', 'Fosfor Seviyesi (mg/kg)',
              'Potasyum Seviyesi (mg/kg)', 'Ortam Sıcaklığı (°C)',
              'Nem (%)', 'Işık Şiddeti (Lux)', 'Klorofil İçeriği (mg/m²)',
              'Elektrokimyasal Sinyal (mV)']

features = ['Soil_Moisture', 'Soil_Temperature', 'Soil_pH', 'Nitrogen_Level',
            'Phosphorus_Level', 'Potassium_Level', 'Ambient_Temperature',
            'Humidity', 'Light_Intensity', 'Chlorophyll_Content',
            'Electrochemical_Signal']

# --- ARAYÜZ ---
pencere = tk.Tk()
pencere.title("🌱 Akıllı Bitki Analiz Sistemi")
pencere.geometry("1100x700")
pencere.configure(bg="#f4f4f4")

ana_panel = ttk.Frame(pencere)
ana_panel.pack(fill="both", expand=True, padx=20, pady=20)

# Sol Panel (Girdiler)
sol_panel = ttk.LabelFrame(ana_panel, text=" Sensör Verileri ", padding=15)
sol_panel.pack(side="left", fill="both", expand=False)

girdi_kutulari = {}
for i, ozellik in enumerate(ozellikler):
    ttk.Label(sol_panel, text=ozellik, font=("Arial", 9)).grid(row=i, column=0, sticky="w", pady=6)
    entry = ttk.Entry(sol_panel, width=12)
    entry.grid(row=i, column=1, pady=6, padx=10)
    girdi_kutulari[ozellik] = entry

# Sağ Panel (Sonuçlar)
sag_panel = ttk.LabelFrame(ana_panel, text=" Analiz Sonuçları ", padding=15)
sag_panel.pack(side="right", fill="both", expand=True, padx=15)

sonuc_baslik = ttk.Label(sag_panel, text="Değerleri girip 'Analiz Et' butonuna basın.", 
                         font=("Arial", 12, "bold"), foreground="#333", wraplength=400)
sonuc_baslik.pack(pady=10)

grafik_frame = ttk.Frame(sag_panel)
grafik_frame.pack(fill="both", expand=True)

def analiz_et():
    try:
        # 1. Veri Okuma
        veriler = []
        for ozellik in ozellikler:
            val = girdi_kutulari[ozellik].get().strip().replace(',', '.')
            if not val:
                messagebox.showwarning("Eksik Veri", f"Lütfen '{ozellik}' alanını doldurun.")
                return
            veriler.append(float(val))

        # 2. Hazırlık ve Tahmin
        # DİKKAT: DataFrame sütun isimleri eğitimdekiyle birebir aynı olmalı
        df_input = pd.DataFrame([veriler], columns=features)
        
        # Scaler ile dönüştür (Eğitimdeki matematiği uygula)
        X_input_scaled = scaler.transform(df_input)
        
        # Tahmin yap
        ham_tahmin = model.predict(X_input_scaled, verbose=0)[0]
        yuzdeler = ham_tahmin * 100
        
        # En yüksek sınıfı bul
        max_idx = np.argmax(ham_tahmin)
        kazanan_sinif_ing = label_encoder.classes_[max_idx]
        
        # Çeviri
        sozluk = {
            "Healthy": "SAĞLIKLI (Büyür)",
            "High Stress": "YÜKSEK STRES (Büyümez)",
            "Moderate Stress": "ORTA STRES (Zayıf Büyür)"
        }
        kazanan_tr = sozluk.get(kazanan_sinif_ing, kazanan_sinif_ing)
        
        # Rengi ayarla
        renk = "green" if "Healthy" in kazanan_sinif_ing else ("red" if "High" in kazanan_sinif_ing else "orange")
        sonuc_baslik.config(text=f"Sonuç: {kazanan_tr}\nKesinlik: %{yuzdeler[max_idx]:.1f}", foreground=renk)

        # 3. Grafik Çizimi (Yazı çakışmasını engelleyen Legend yöntemi)
        for widget in grafik_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Türkçe Etiketler
        labels_tr = [sozluk.get(l, l) for l in label_encoder.classes_]
        
        # Pasta Grafiği
        wedges, texts, autotexts = ax.pie(
            yuzdeler, 
            autopct=lambda p: f'%{p:.1f}' if p > 0 else '', # Sadece %0'dan büyükleri yaz
            startangle=90,
            colors=['#66b3ff', '#ff9999', '#ffcc99'],
            textprops=dict(color="black")
        )
        
        ax.legend(wedges, labels_tr, title="Durumlar", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        ax.set_title("Olasılık Dağılımı")
        
        canvas = FigureCanvasTkAgg(fig, master=grafik_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    except Exception as e:
        messagebox.showerror("Hata", f"Beklenmedik bir hata:\n{e}")

# Buton
btn = ttk.Button(sol_panel, text="🔍 ANALİZ ET", command=analiz_et)
btn.grid(row=len(ozellikler)+1, column=0, columnspan=2, pady=20, ipady=5)

pencere.mainloop()