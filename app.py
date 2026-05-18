import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================
# KONFIGURASI HALAMAN
# ==============================

st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide"
)

# ==============================
# FITUR MODEL
# ==============================

MODEL_FEATURES = [
    "Engine_size",
    "Horsepower",
    "Wheelbase",
    "Width",
    "Length",
    "Curb_weight",
    "Fuel_capacity",
    "Fuel_efficiency"
]

TARGET = "Price_in_thousands"


# ==============================
# LOAD DATA DAN TRAIN MODEL
# ==============================

@st.cache_resource
def train_model():
    # Membaca dataset
    df = pd.read_excel("Car_sales.xlsx")

    # Membuat salinan data
    df_model = df.copy()

    # Pastikan fitur dan target berbentuk numerik
    for col in MODEL_FEATURES + [TARGET]:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    # Hapus data dengan target kosong
    df_model = df_model.dropna(subset=[TARGET])

    # Memisahkan X dan y
    X = df_model[MODEL_FEATURES]
    y = df_model[TARGET]

    # Split data 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Pipeline model
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", LinearRegression())
    ])

    # Training model
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Koefisien model
    regressor = model.named_steps["regressor"]
    coefficient_df = pd.DataFrame({
        "Feature": MODEL_FEATURES,
        "Coefficient": regressor.coef_
    }).sort_values(by="Coefficient", ascending=False)

    result = {
        "model": model,
        "features": MODEL_FEATURES,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mse": mse,
        "coefficient_df": coefficient_df,
        "intercept": regressor.intercept_,
        "data_shape": df_model.shape,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0]
    }

    return result


model_bundle = train_model()

model = model_bundle["model"]
model_features = model_bundle["features"]
rmse = model_bundle["rmse"]
mae = model_bundle["mae"]
r2_score_value = model_bundle["r2"]
mse = model_bundle["mse"]
coefficient_df = model_bundle["coefficient_df"]
intercept = model_bundle["intercept"]


# ==============================
# HEADER
# ==============================

st.title("🚗 Prediksi Harga Mobil Menggunakan Linear Regression")

st.write(
    """
    Aplikasi ini digunakan untuk memprediksi harga mobil berdasarkan spesifikasi kendaraan.
    Model yang digunakan adalah **Linear Regression** dan dilatih langsung dari dataset
    `Car_sales.xlsx`.
    """
)

st.info(
    "Masukkan spesifikasi mobil, lalu klik tombol prediksi untuk mendapatkan estimasi harga."
)


# ==============================
# LAYOUT UTAMA
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
# HASIL PREDIKSI
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

        lower_bound_dollar = max(0, (predicted_price_thousand - rmse) * 1000)
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
            st.dataframe(input_data, use_container_width=True)

    else:
        st.write("Masukkan spesifikasi mobil, lalu klik tombol **Hitung Harga Mobil**.")


# ==============================
# INFORMASI MODEL
# ==============================

st.divider()

st.subheader("Informasi Model")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric("RMSE", f"{rmse:.4f}")
metric_col2.metric("MAE", f"{mae:.4f}")
metric_col3.metric("R2 Score", f"{r2_score_value:.4f}")
metric_col4.metric("MSE", f"{mse:.4f}")

st.write("### Detail Data")
st.write(f"Jumlah data setelah cleaning: **{model_bundle['data_shape'][0]}**")
st.write(f"Jumlah data training: **{model_bundle['train_size']}**")
st.write(f"Jumlah data testing: **{model_bundle['test_size']}**")

st.write("### Koefisien Linear Regression")
st.dataframe(coefficient_df, use_container_width=True)

st.write(f"**Intercept:** {intercept:.4f}")


# ==============================
# FOOTER
# ==============================

st.divider()

st.write(
    """
    Sistem ini dibuat untuk Final Project Sains Data.
    Model dilatih menggunakan algoritma Linear Regression dengan fitur:
    Engine Size, Horsepower, Wheelbase, Width, Length, Curb Weight,
    Fuel Capacity, dan Fuel Efficiency.
    """
)
