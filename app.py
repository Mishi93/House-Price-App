import streamlit as st
import joblib
import numpy as np

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        font-size: 18px;
        color: #7f8c8d;
    }
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        background: linear-gradient(135deg, #4CAF50, #2ecc71);
        color: white;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("house_price_model.joblib")

# ----------------------------
# Header
# ----------------------------
st.markdown('<p class="title">🏠 House Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter property details to estimate price instantly</p>', unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🏡 Property Details")

area = st.sidebar.number_input("Area (sq ft)", 300, 10000)
bedrooms = st.sidebar.number_input("Bedrooms", 1, 10)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 10)
stories = st.sidebar.number_input("Stories", 1, 5)
parking = st.sidebar.number_input("Parking Spaces", 0, 5)

binary_map = {"No": 0, "Yes": 1}

mainroad = binary_map[st.sidebar.selectbox("Main Road", ["No", "Yes"])]
guestroom = binary_map[st.sidebar.selectbox("Guest Room", ["No", "Yes"])]
basement = binary_map[st.sidebar.selectbox("Basement", ["No", "Yes"])]
hotwaterheating = binary_map[st.sidebar.selectbox("Hot Water Heating", ["No", "Yes"])]
airconditioning = binary_map[st.sidebar.selectbox("Air Conditioning", ["No", "Yes"])]
prefarea = binary_map[st.sidebar.selectbox("Preferred Area", ["No", "Yes"])]

furnishing_map = {
    "Unfurnished": 0,
    "Semi-Furnished": 1,
    "Furnished": 2
}

furnishingstatus = furnishing_map[
    st.sidebar.selectbox("Furnishing Status",
                         ["Unfurnished", "Semi-Furnished", "Furnished"])
]

# ----------------------------
# Main Layout Columns
# ----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.image(
        "https://images.unsplash.com/photo-1568605114967-8130f3a36994",
            width=1000
    )

with col2:
    st.subheader("📊 Property Summary")
    st.write(f"**Area:** {area} sq ft")
    st.write(f"**Bedrooms:** {bedrooms}")
    st.write(f"**Bathrooms:** {bathrooms}")
    st.write(f"**Stories:** {stories}")
    st.write(f"**Parking:** {parking}")

# ----------------------------
# Prediction
# ----------------------------
st.markdown("##")

if st.button("🔮 Predict Price"):

    input_data = np.array([[
        area, bedrooms, bathrooms, stories, parking,
        mainroad, guestroom, basement,
        hotwaterheating, airconditioning,
        prefarea, furnishingstatus
    ]])

    prediction = model.predict(input_data)[0]

    st.markdown(
        f'<div class="prediction-box">💰 Estimated Price: ${prediction:,.2f}</div>',
        unsafe_allow_html=True
    )