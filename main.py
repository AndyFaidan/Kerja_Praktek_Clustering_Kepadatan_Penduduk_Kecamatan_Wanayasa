import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from streamlit_option_menu import option_menu


def local_css(file_name):
    with open(file_name, 'r') as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("style.css")

# Memuat data dari CSV
file_path = "original_table.csv"
df = pd.read_csv(file_path)

# Sidebar
st.sidebar.title("Pengaturan Klaster")

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Data Asli", "Tabel Klaster", "Visualisasi Data", "Peta Folium", "Sillhoute Score"],
        icons=["journal-code", "journal-check","graph-up", "pin-map", "activity"],
        menu_icon="cast",
        default_index=0,
    )

    st.markdown("----")

num_clusters = st.sidebar.slider("Jumlah Klaster", 2, 10, 3)

# Kolom untuk pengelompokan (2011-2022)
kolom_pengelompokan = [str(tahun) for tahun in range(2011, 2020)]

# Memastikan tidak ada nilai None di dalam kolom_pengelompokan
df[kolom_pengelompokan] = df[kolom_pengelompokan].astype(float)  # Konversi ke tipe data float

# Kategorikan klaster berdasarkan rentang kepadatan penduduk
population_density_ranges = [1300, 2280, 3900, np.inf]
df["Density Category"] = pd.cut(df[kolom_pengelompokan].mean(axis=1), bins=population_density_ranges, labels=["Tidak Padat", "Padat", "Sangat Padat"])

# Metode Elbow untuk menentukan jumlah klaster yang optimal
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(df[kolom_pengelompokan])
    inertia_values.append(kmeans.inertia_)

# Menampilkan plot Metode Elbow
fig, ax = plt.subplots()
ax.plot(range(1, 11), inertia_values, marker='o')
ax.set_xlabel('Jumlah Klaster')
ax.set_ylabel('Inertia (Within-cluster Sum of Squares)')
st.sidebar.pyplot(fig)

# Konten utama
st.markdown('<h1 class="centered-title">Analisis Klaster Kepadatan Penduduk</h1>', unsafe_allow_html=True)
st.markdown("----")

# Menampilkan informasi Metode Elbow
st.sidebar.write("### Informasi Metode Elbow:")
st.sidebar.write("Metode Elbow membantu menentukan jumlah klaster optimal dengan melihat titik di mana penurunan inersia tidak lagi signifikan.")

# Klaster dengan KMeans (gunakan jumlah klaster yang telah ditentukan)
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df["Cluster"] = kmeans.fit_predict(df[kolom_pengelompokan])

# Menghitung skor siluet untuk setiap baris
df["Silhouette Score"] = silhouette_samples(df[kolom_pengelompokan], df["Cluster"])

# Menghitung skor siluet untuk seluruh dataset
silhouette_avg = silhouette_score(df[kolom_pengelompokan], df["Cluster"])

# Menampilkan hasil klasterisasi
if selected == "Data Asli":
    # Menampilkan tabel data asli dengan skor siluet
    st.write(f"### {selected}:")
    st.write(df)

elif selected == "Tabel Klaster":
    st.write(f"### {selected}:")
    # Menampilkan tabel klaster
    for cluster_id in range(num_clusters):
        cluster_df = df[df['Cluster'] == cluster_id]
        density_category = cluster_df['Density Category'].iloc[0]
        st.write(f"### Tabel Klaster {cluster_id + 0}  ({density_category})")
        st.write(cluster_df)

elif selected == "Visualisasi Data":
    st.write(f"### {selected}:")
    ringkasan_klaster = df.groupby("Cluster").agg({
        "Desa": "count",
        **{tahun: "mean" for tahun in kolom_pengelompokan}
    }).rename(columns={"Desa": "Jumlah Desa", **{tahun: f"Rata-rata Populasi {tahun}" for tahun in kolom_pengelompokan}})
    st.write(ringkasan_klaster)

    st.markdown("---")

    # Grafik Scatter Plot Latitude dan Longitude sesuai dengan Klaster
    st.write("### Scatter Plot Latitude dan Longitude:")
    st.write("Grafik ini menunjukkan persebaran desa pada peta berdasarkan klaster.")

    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_id in range(num_clusters):
        cluster_data = df[df['Cluster'] == cluster_id]
        ax.scatter(cluster_data['Longitude'], cluster_data['Latitude'], label=f'Cluster {cluster_id + 1}')

    ax.set_xlabel('Garis Bujur (Longitude)')
    ax.set_ylabel('Garis Lintang (Latitude)')
    ax.set_title('Scatter Plot Latitude dan Longitude')
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")

    # Grafik Garis Pertumbuhan Penduduk untuk Setiap Klaster
    st.write("### Grafik Garis Pertumbuhan Penduduk untuk Setiap Klaster:")
    st.write("Grafik ini menunjukkan rata-rata pertumbuhan penduduk setiap klaster dari tahun 2012 hingga 2021.")

    plt.figure(figsize=(10, 5))
    for cluster_id in range(num_clusters):
        cluster_data = df[df['Cluster'] == cluster_id]
        plt.plot(kolom_pengelompokan, cluster_data[kolom_pengelompokan].mean(), label=f'Cluster {cluster_id + 1}')

    plt.xlabel("Tahun")
    plt.ylabel("Rata-rata Populasi")
    plt.title("Pertumbuhan Penduduk Setiap Klaster")
    plt.legend()
    st.pyplot(plt)

# Peta Folium
elif selected == "Peta Folium":
    st.write(f"### {selected}:")

    # Membuat peta dengan lokasi rata-rata garis lintang dan garis bujur
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

    # Menambahkan marker cluster
    marker_cluster = MarkerCluster().add_to(m)

    # Menambahkan penanda untuk setiap klaster
    for i, row in df.iterrows():
        # Dapatkan kategori kepadatan berdasarkan rentang kepadatan penduduk yang diperbarui
        density = pd.cut([row[k] for k in kolom_pengelompokan], bins=population_density_ranges, labels=["Tidak Padat", "Padat", "Sangat Padat"])

        # Menambahkan check untuk memastikan density tidak kosong
        if not density.isna().all():
            # Mengambil nilai pertama dari objek Categorical atau menggunakan nilai default jika tidak ada nilai
            density_value = density[0] if not density.isna().all() else None

            # Menyesuaikan warna ikon berdasarkan tingkat kepadatan penduduk
            icon_color = 'red' if density_value == 'Sangat Padat' else 'orange' if density_value == 'Padat' else 'green'

            folium.Marker([row['Latitude'], row['Longitude']],
                          popup=f"Klaster {row['Cluster'] + 1}: {row['Desa']}<br>Kepadatan: {density_value}",
                          icon=folium.Icon(color=icon_color)).add_to(marker_cluster)

    # Menambahkan peta ke Streamlit
    folium_static(m)


    # Menampilkan keterangan data pada peta
    st.write("Keterangan Data pada Peta:")
    st.write("- Setiap penanda menunjukkan lokasi desa dalam klaster tertentu.")
    st.write("- Warna penanda mengindikasikan tingkat kepadatan penduduk: Merah (Sangat Padat), Orange (Padat), Hijau (Tidak Padat).")
    st.write("- Klik pada penanda untuk melihat informasi lebih lanjut, termasuk klaster dan kepadatan penduduk.")

if selected == "Sillhoute Score":
    # Menampilkan tabel data asli dengan skor siluet
    st.write(f"### {selected}:")

    # Data yang digunakan (X_scaled adalah data yang sudah dinormalisasi)
    X_scaled = df.drop(['Desa', 'Latitude', 'Longitude', 'Cluster', 'Silhouette Score', 'Density Category'], axis=1)

    # List untuk menyimpan nilai silhouette score
    silhouette_scores = []

    # Range jumlah klaster yang akan dicoba
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=50)
        kmeans.fit(X_scaled)

        # Mendapatkan label klaster untuk setiap data point
        cluster_labels = kmeans.labels_

        # Menghitung silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)

        # Menyimpan nilai silhouette score
        silhouette_scores.append(silhouette_avg)

        print("For n_clusters = {0}, the silhouette score is {1:.2f}".format(num_clusters, silhouette_avg))

    # Plot Silhouette Score
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, marker='o', linestyle='-', color='b')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    st.pyplot(plt)

    # Menampilkan keterangan hasil Silhouette Score
    st.write("### Hasil Silhouette Score:")
    st.write("Silhouette Score mengukur seberapa baik setiap objek dalam klaster yang diberikan sesuai dengan klaster lainnya. "
             "Nilai Silhouette Score berkisar dari -1 (klaster yang salah) hingga 1 (klaster yang baik).")

    for num_clusters, silhouette_avg in zip(range_n_clusters, silhouette_scores):
        st.write(f"For n_clusters = {num_clusters}, the silhouette score is {silhouette_avg:.2f}")

    st.write("Dari hasil tersebut, nilai tertinggi pada jumlah klaster 3 dengan Silhouette Score sekitar 0.60. "
             "Nilai tinggi menandakan bahwa klaster tersebut memiliki objek yang lebih serupa satu sama lain dan berbeda dari klaster lainnya, "
             "menunjukkan bahwa pengelompokan menjadi 3 klaster adalah pilihan yang baik.")


st.markdown("---")
st.sidebar.write("### Kesimpulan:")
# Menyimpan hasil klaster untuk setiap klaster
cluster_results = []
for cluster_id in range(num_clusters):
    cluster_data = df[df['Cluster'] == cluster_id]
    
    # Memeriksa apakah ada data di dalam klaster
    if not cluster_data.empty:
        # Menggabungkan desa menjadi satu teks
        desa_text = ', '.join(cluster_data['Desa'].tolist())
        
        # Memeriksa apakah ada data kepadatan penduduk di klaster
        if not cluster_data['Density Category'].empty:
            density_value = cluster_data['Density Category'].iloc[0]
        else:
            density_value = "Tidak ada data"
        
        cluster_results.append({
            f"Klaster {cluster_id + 1}": desa_text,
            f"Kepadatan Penduduk Klaster {cluster_id + 1}": density_value
        })
    else:
        cluster_results.append({
            f"Klaster {cluster_id + 1}": "Tidak ada data",
            f"Kepadatan Penduduk Klaster {cluster_id + 1}": "Tidak ada data"
        })

# Menampilkan kesimpulan
for result in cluster_results:
    for key, value in result.items():
        st.sidebar.write(f"{key}: {value}")

# Menambahkan penjelasan kesimpulan berdasarkan hasil klasterisasi, nilai Silhouette Score, dll.
if silhouette_avg >= 0.5:  # Adjust the threshold as needed
    conclusion_message = f"Penggunaan {num_clusters} klaster terlihat optimal, dengan nilai Silhouette Score rata-rata sebesar {silhouette_avg:.2f}. " \
                         f"Nilai Silhouette Score yang tinggi menunjukkan bahwa objek dalam satu klaster memiliki kesamaan yang tinggi dan perbedaan yang rendah, " \
                         f"mengindikasikan hasil klasterisasi yang baik."
else:
    conclusion_message = f"Penggunaan {num_clusters} klaster tidak optimal, karena nilai Silhouette Score rata-rata hanya sebesar {silhouette_avg:.2f}. " \
                         f"Silhouette Score yang tinggi menunjukkan bahwa objek dalam satu klaster memiliki kesamaan yang tinggi dan perbedaan yang rendah, " \
                         f"sehingga nilai yang rendah dapat mengindikasikan penyebaran atau overlap yang tinggi antar klaster."

st.sidebar.write(conclusion_message)


# Informasi Tambahan
st.sidebar.markdown("---")
st.sidebar.write("### Informasi Tambahan:")
st.sidebar.write("Analisis ini didasarkan pada data kepadatan penduduk di tiap desa. Metode klasterisasi membantu mengelompokkan desa-desa dengan karakteristik serupa, memberikan pemahaman yang lebih baik tentang pola kepadatan penduduk di wilayah tersebut.")
st.sidebar.write("Selain itu, Metode Elbow digunakan untuk menentukan jumlah klaster optimal. Jumlah klaster yang dipilih didasarkan pada poin di mana penurunan inersia tidak lagi signifikan, "
                 "sehingga memberikan klasterisasi yang baik. Analisis ini juga mempertimbangkan tingkat kepadatan penduduk dalam pengelompokan desa.")
