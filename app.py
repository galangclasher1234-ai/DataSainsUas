import streamlit as st
import pandas as pd
import joblib

# ==============================
# KONFIGURASI HALAMAN
# ==============================

st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide"
)

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    model_bundle = joblib.load("car_price_model.joblib")
    return model_bundle

model_bundle = load_model()

model = model_bundle["model"]
model_features = model_bundle["features"]
rmse = model_bundle["rmse"]
mae = model_bundle["mae"]
r2_score = model_bundle["r2_score"]

# ==============================
# HEADER
# ==============================

st.title("🚗 Prediksi Harga Mobil Menggunakan Linear Regression")
st.write(
    """
    Aplikasi ini digunakan untuk memprediksi harga mobil berdasarkan spesifikasi kendaraan.
    Masukkan spesifikasi mobil pada form di bawah, lalu klik tombol prediksi untuk melihat estimasi harga.
    """
)

st.info(
    "Catatan: hasil prediksi merupakan estimasi berdasarkan model Linear Regression dan dataset Car_sales."
)

# ==============================
# LAYOUT
# ==============================

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Input Spesifikasi Mobil")

    engine_size = st.number_input(
        "Engine Size",
        min_value=0.5,
        max_value=10.0,
        value=2.5,
        step=0.1
    )

    horsepower = st.number_input(
        "Horsepower",
        min_value=50,
        max_value=500,
        value=160,
        step=5
    )

    wheelbase = st.number_input(
        "Wheelbase",
        min_value=80.0,
        max_value=140.0,
        value=106.0,
        step=0.5
    )

    width = st.number_input(
        "Width",
        min_value=50.0,
        max_value=90.0,
        value=70.0,
        step=0.5
    )

    length = st.number_input(
        "Length",
        min_value=120.0,
        max_value=250.0,
        value=185.0,
        step=0.5
    )

    curb_weight = st.number_input(
        "Curb Weight",
        min_value=1.0,
        max_value=6.0,
        value=3.1,
        step=0.1
    )

    fuel_capacity = st.number_input(
        "Fuel Capacity",
        min_value=5.0,
        max_value=40.0,
        value=16.0,
        step=0.5
    )

    fuel_efficiency = st.number_input(
        "Fuel Efficiency",
        min_value=5,
        max_value=60,
        value=28,
        step=1
    )

    predict_button = st.button("Hitung Harga Mobil")

# ==============================
# PREDIKSI
# ==============================

with right_col:
    st.subheader("Hasil Prediksi")

    input_data = pd.DataFrame({
        "Engine_size": [engine_size],
        "Horsepower": [horsepower],
        "Wheelbase": [wheelbase],
        "Width": [width],
        "Length": [length],
        "Curb_weight": [curb_weight],
        "Fuel_capacity": [fuel_capacity],
        "Fuel_efficiency": [fuel_efficiency]
    })

    input_data = input_data[model_features]

    if predict_button:
        predicted_price_thousand = model.predict(input_data)[0]
        predicted_price_dollar = predicted_price_thousand * 1000

        lower_bound_dollar = (predicted_price_thousand - rmse) * 1000
        upper_bound_dollar = (predicted_price_thousand + rmse) * 1000

        if predicted_price_dollar < 0:
            st.error(
                "Prediksi menghasilkan nilai negatif. "
                "Silakan cek kembali kombinasi spesifikasi yang dimasukkan."
            )
        else:
            st.metric(
                label="Perkiraan Harga Mobil",
                value=f"${predicted_price_dollar:,.2f}"
            )

            st.write("### Rentang Estimasi Berdasarkan RMSE")
            st.write(
                f"${lower_bound_dollar:,.2f} sampai ${upper_bound_dollar:,.2f}"
            )

            st.write("### Data Input")
            st.dataframe(input_data)

            st.write("### Informasi Model")
            metric_df = pd.DataFrame({
                "Metric": ["RMSE", "MAE", "R2 Score"],
                "Value": [rmse, mae, r2_score]
            })

            st.dataframe(metric_df)

    else:
        st.write("Masukkan spesifikasi mobil, lalu klik tombol **Hitung Harga Mobil**.")

# ==============================
# FOOTER
# ==============================

st.divider()

st.write(
    """
    Sistem ini dibuat untuk Final Project Sains Data.
    Model yang digunakan adalah Linear Regression dengan fitur:
    Engine Size, Horsepower, Wheelbase, Width, Length, Curb Weight,
    Fuel Capacity, dan Fuel Efficiency.
    """
)
