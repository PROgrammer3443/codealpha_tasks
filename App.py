import streamlit as st # type: ignore
import numpy as np # type: ignore
from PIL import Image, ImageOps # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from streamlit_drawable_canvas import st_canvas # type: ignore

model = load_model("mnist_model.h5", compile=False)

def preprocess(img):
    img = img.convert("L")
    arr = np.array(img)
    if np.mean(arr) > 127:
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img).reshape(1, 28, 28, 1).astype("float32") / 255.0
    return img

st.title("Digit Recognition")

tab1, tab2 = st.tabs(["Draw", "Upload"])

with tab1:
    canvas = st_canvas(
        fill_color="black",
        stroke_width=8,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    if st.button("Predict Drawing"):
        if canvas.image_data is not None:
            img = Image.fromarray(np.uint8(canvas.image_data))
            img = ImageOps.invert(img.convert("L"))
            arr = preprocess(img)
            pred = np.argmax(model.predict(arr))
            st.subheader(str(pred))

with tab2:
    file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])
    if file:
        img = Image.open(file)
        st.image(img, width=150)
        arr = preprocess(img)
        if st.button("Predict Upload"):
            pred = np.argmax(model.predict(arr))
            st.subheader(str(pred))
