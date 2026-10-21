import json
import numpy as np
import streamlit as st

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠")


@st.cache_resource
def load_model():
    with open("model.json") as f:
        return json.load(f)


model = load_model()
features = model["features"]
coef = np.array(model["coef"])

st.title("🏠 California House Price Predictor")
st.write(
    "Multiple linear regression built from scratch with the normal equation. "
    "Enter the details of a housing block and get the predicted median house value."
)

values = []
for i, name in enumerate(features):
    lo = float(model["feature_min"][i])
    hi = float(model["feature_max"][i])
    default = float(model["feature_median"][i])
    label = name.replace("_", " ").title()
    values.append(st.number_input(label, min_value=lo, max_value=hi, value=default))

if st.button("Predict price"):
    prediction = float(np.dot(np.array(values), coef) + model["intercept"])
    st.success(f"Predicted median house value: ${prediction:,.0f}")
    st.caption("Estimate from a simple linear model; real prices vary.")
