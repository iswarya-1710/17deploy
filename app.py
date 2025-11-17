import streamlit as st
import numpy as np
import joblib
model = joblib.load("model.joblib")
st.title("Customer Satisfaction")
st.write("Enter the metrics to predict customer satisfaction")

order_price = st.number_input("order price", min_value=585,max_value=35180,step=1)
delivery_charges  = st.number_input("deliverycharges", min_value=46.35, max_value=107.58,step=0.1)
customer_long = st.number_input("customerlong", min_value=144.9272705,max_value=145.0198375, step=0.01)
coupon_discount = st.number_input("coupondiscount", min_value=0, max_value=20,step=1)
order_total = st.number_input("ordertotal", min_value=639.29,max_value=33947.06, step=0.1)
distance_to_nearest_warehouse = st.number_input("distance_to_nearest_warehouse", min_value=0.1078,max_value=2.2493, step=0.1)
if st.button("Predict"):
  features = np.array([[order_price, delivery_charges, customer_long,coupon_discount,order_total, distance_to_nearest_warehouse]])
  prediction = model.predict(features)[0]
  st.success(f" Predicted: {prediction}")
