
import pickle
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Real Estate Inference", layout="wide")
st.title("Real Estate Price & Class Inference")
st.caption("Uses regression_model.pkl, classification_model.pkl, preprocessing_artifacts.pkl")


@st.cache_resource
def load_artifacts():
    with open("regression_model.pkl", "rb") as file_handle:
        reg_model = pickle.load(file_handle)
    with open("classification_model.pkl", "rb") as file_handle:
        cls_model = pickle.load(file_handle)
    with open("preprocessing_artifacts.pkl", "rb") as file_handle:
        prep = pickle.load(file_handle)
    return reg_model, cls_model, prep


@st.cache_data
def load_dataset_values():
    data = pd.read_csv("dataset_clean.csv", index_col=0)
    values = {
        "location": sorted(data["location"].dropna().astype(str).unique().tolist()) if "location" in data.columns else ["Unknown"],
        "city": sorted(data["city"].dropna().astype(str).unique().tolist()) if "city" in data.columns else ["Unknown"],
        "governorate": sorted(data["governorate"].dropna().astype(str).unique().tolist()) if "governorate" in data.columns else ["Unknown"],
        "age": sorted(data["age"].dropna().astype(str).unique().tolist()) if "age" in data.columns else ["1-5"],
    }
    return values


def encode_binary(label, default=0):
    return int(st.sidebar.selectbox(label, [0, 1], index=default))


try:
    reg_model, cls_model, prep = load_artifacts()
except FileNotFoundError as error:
    st.error(f"Missing file: {error}")
    st.stop()

category_values = load_dataset_values()

st.sidebar.header("Property Inputs")

with st.sidebar.form("inference_form"):
    area = st.number_input("Area", min_value=10.0, value=300.0)
    room = st.number_input("Rooms", min_value=1.0, value=4.0)
    bathroom = st.number_input("Bathrooms", min_value=1.0, value=2.0)
    pieces = st.number_input("Pieces", min_value=1.0, value=8.0)
    latt = st.number_input("Latitude", value=36.8)
    longi = st.number_input("Longitude", value=10.2)
    distance_to_capital = st.number_input("Distance to capital", min_value=0.0, value=20.0)
    state = st.number_input("State", min_value=0.0, value=1.0)

    garage = encode_binary("Garage", default=1)
    garden = encode_binary("Garden")
    concierge = encode_binary("Concierge")
    beach_view = encode_binary("Beach view")
    mountain_view = encode_binary("Mountain view")
    pool = encode_binary("Pool")
    elevator = encode_binary("Elevator", default=1)
    furnished = encode_binary("Furnished")
    equipped_kitchen = encode_binary("Equipped kitchen", default=1)
    central_heating = encode_binary("Central heating", default=1)
    air_conditioning = encode_binary("Air conditioning", default=1)

    location = st.selectbox("Location", category_values["location"], index=0)
    city = st.selectbox("City", category_values["city"], index=0)
    governorate = st.selectbox("Governorate", category_values["governorate"], index=0)
    age = st.selectbox("Age category", category_values["age"], index=0)

    submitted = st.form_submit_button("Predict")


if submitted:
    row = {
        "Area": area,
        "room": room,
        "bathroom": bathroom,
        "pieces": pieces,
        "latt": latt,
        "long": longi,
        "distance_to_capital": distance_to_capital,
        "state": state,
        "garage": garage,
        "garden": garden,
        "concierge": concierge,
        "beach_view": beach_view,
        "mountain_view": mountain_view,
        "pool": pool,
        "elevator": elevator,
        "furnished": furnished,
        "equipped_kitchen": equipped_kitchen,
        "central_heating": central_heating,
        "air_conditioning": air_conditioning,
        "location": location,
        "city": city,
        "governorate": governorate,
        "age": age,
    }

    x_raw = pd.DataFrame([row])

    x_num = x_raw.select_dtypes(include=[np.number]).copy()
    if {"latt", "long"}.issubset(x_num.columns):
        x_num["lat_long_interaction"] = x_num["latt"] * x_num["long"]
    if {"Area", "room"}.issubset(x_num.columns):
        x_num["area_per_room"] = x_num["Area"] / (x_num["room"] + 1)

    for col in prep.get("cat_cols", []):
        mapping = prep.get("te_maps", {}).get(col, {})
        value = x_raw[col].iloc[0] if col in x_raw.columns else "missing"
        x_num[f"{col}_te"] = [mapping.get(value, prep.get("global_mean", 0.0))]

    for col in prep["feature_columns"]:
        if col not in x_num.columns:
            x_num[col] = 0.0

    x_model = x_num[prep["feature_columns"]]

    price_pred = float(reg_model.predict(x_model)[0])
    cls_pred = int(cls_model.predict(x_model)[0])
    cls_label = "High price class" if cls_pred == 1 else "Low price class"

    st.subheader("Predictions")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Price (TND)", f"{price_pred:,.0f}")
    col2.metric("Predicted Class", cls_label)

    if hasattr(cls_model, "predict_proba"):
        proba = cls_model.predict_proba(x_model)[0]
        st.write(f"Class probability (Low/High): {proba[0]:.3f} / {proba[1]:.3f}")
