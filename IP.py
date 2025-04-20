import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_csv("destinasi-wisata-indonesia.csv")

# Preprocessing
data = df.copy()
data = data[['Place_Name', 'Category', 'City', 'Price', 'Rating', 'Rating_Count', 'Time_Minutes']]

# Tangani missing values
data['Time_Minutes'] = data['Time_Minutes'].fillna(data['Time_Minutes'].median())

# Konversi rating dari skala 1–50 ke 0–5
data['Rating'] = data['Rating'] / 10.0

# Encode kolom kategorikal
le_kategori = LabelEncoder()
le_lokasi = LabelEncoder()
data['Category'] = le_kategori.fit_transform(data['Category'])
data['City'] = le_lokasi.fit_transform(data['City'])

# Fitur yang digunakan
fitur = ['Category', 'City', 'Price', 'Rating', 'Rating_Count', 'Time_Minutes']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[fitur])

# Latih model KNN
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(data_scaled)

# Streamlit UI
st.title("🎯 Sistem Rekomendasi Destinasi Wisata")

kategori_list = df['Category'].unique().tolist()
lokasi_list = df['City'].unique().tolist()

kategori_input = st.selectbox("Pilih Kategori", kategori_list)
lokasi_input = st.selectbox("Pilih Lokasi", lokasi_list)
harga_input = st.number_input("Harga Maksimal (Rp)", min_value=0, value=50000)
rating_input = st.slider("Rating Minimal", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
jumlah_rating_input = st.number_input("Jumlah Rating", min_value=0, value=20)
waktu_input = st.slider("Estimasi Waktu Kunjungan (menit)", min_value=10.0, max_value=600.0, value=120.0, step=10.0)

if st.button("Rekomendasikan"):
    # Encode input user
    kategori_encoded = le_kategori.transform([kategori_input])[0]
    lokasi_encoded = le_lokasi.transform([lokasi_input])[0]
    input_user = [[kategori_encoded, lokasi_encoded, harga_input, rating_input, jumlah_rating_input, waktu_input]]
    input_scaled = scaler.transform(input_user)

    # Cari rekomendasi
    distances, indices = knn.kneighbors(input_scaled)

    st.subheader("✨ Rekomendasi Destinasi:")
    for idx in indices[0]:
        row = df.iloc[idx]
        st.markdown(f"**{row['Place_Name']}**")
        st.markdown(f"- Kategori: {row['Category']}")
        st.markdown(f"- Lokasi: {row['City']}")
        st.markdown(f"- Harga: Rp{row['Price']:,}")
        st.markdown(f"- Rating: {row['Rating'] / 10:.1f}")
        st.markdown(f"- Estimasi Waktu: {row['Time_Minutes']} menit")
        st.markdown("---")
