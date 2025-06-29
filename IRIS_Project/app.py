import streamlit as st
from controller.LoadModel import LoadModel
from controller.GetPrediction import GetPrediction

st.title("🌸 Iris Flower Prediction App")
st.write("This app predicts the species of Iris flower based on its features.")


sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0)


if st.button("🔍 Predict"):
    model = LoadModel("model/iris_model.pkl")
    data = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = GetPrediction(model, data)
    st.success(f"The predicted species is: *{prediction}*")


st.markdown("---")
st.write("Made with ❤ by Omar Atef")

st.markdown(
    """
    <div style="display: flex; gap: 10px; align-items: center;">
        <a href="https://github.com/o2204" target="_blank">
            <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" alt="GitHub"/>
        </a>
        <a href="https://www.kaggle.com/omaratef200" target="_blank">
            <img src="https://img.icons8.com/ios-filled/30/000000/linkedin.png" alt="LinkedIn"/>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)