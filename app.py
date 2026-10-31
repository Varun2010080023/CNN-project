"""Streamlit frontend for the locally trained apple/banana CNN."""
import hashlib
import io
import json
import random
import warnings

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from model import CLASSES, MODEL_PATH, ROOT, load_model, predict

st.set_page_config(page_title="Fruit Lens", page_icon="🍎", layout="centered")


@st.cache_resource
def get_model(modified_at):
    # The timestamp invalidates the cached model after retraining.
    torch.set_num_threads(2)
    return load_model()


def read_fruit_info():
    data = json.loads((ROOT / "fruit_info.json").read_text(encoding="utf-8"))
    for fruit in CLASSES:
        info = data[fruit]
        if not isinstance(info["facts"], list) or not all(isinstance(f, str) and f.strip() for f in info["facts"]):
            raise ValueError(f"{fruit}: facts must be a list of nonempty strings.")
        nutrition = info["nutrition"]
        if not isinstance(nutrition["serving"], str) or not isinstance(nutrition["nutrients"], dict):
            raise ValueError(f"{fruit}: nutrition requires a serving string and nutrients object.")
        if not all(isinstance(k, str) and isinstance(v, (str, int, float)) for k, v in nutrition["nutrients"].items()):
            raise ValueError(f"{fruit}: each nutrient needs a name and a text or numeric value.")
        if nutrition["nutrients"] and not nutrition["serving"].strip():
            raise ValueError(f"{fruit}: specify a serving size for the nutrition values.")
    return data


def show_fruit_details(fruit, info, image_key):
    st.divider()
    st.subheader("A little food for thought")
    facts = info["facts"]
    if facts:
        # Keep the fact stable on reruns; change it for new images or on request.
        key = (image_key, fruit, tuple(facts))
        if st.session_state.get("fact_key") != key:
            st.session_state.fact_key = key
            st.session_state.fact = random.choice(facts)
        if st.button("Give me another fact", disabled=len(set(facts)) < 2):
            st.session_state.fact = random.choice([f for f in facts if f != st.session_state.fact])
        st.info(st.session_state.fact)
    else:
        st.info(f"Random facts for {fruit} have not been added yet.")
    st.subheader("Nutrition")
    nutrition = info["nutrition"]
    if nutrition["nutrients"]:
        st.caption(f"Values for {nutrition['serving']}. These are reference values, not measurements from the photo.")
        st.table({"Nutrient": list(nutrition["nutrients"]), "Amount": [str(v) for v in nutrition["nutrients"].values()]})
        if nutrition.get("source"):
            st.caption(f"Source: {nutrition['source']}")
    else:
        st.info(f"Nutrition information for {fruit} has not been added yet.")


def main():
    st.markdown("<p style='letter-spacing: .18em; font-size: .8rem; font-weight: 700;'>FRUIT LENS</p>", unsafe_allow_html=True)
    st.title("Apple or banana?")
    st.write("Take a closer look at your fruit. Add a photo to discover its name, a fun fact, and nutrition information.")
    st.caption("Use a clear photo of one apple or banana. This model only distinguishes these two fruits and cannot reliably reject other objects.")
    if not MODEL_PATH.exists():
        st.info("The model needs to finish training before you can identify fruit.")
        st.code("python train.py", language="bash")
        return
    try:
        model = get_model(MODEL_PATH.stat().st_mtime_ns)
    except (OSError, RuntimeError, ValueError, KeyError, EOFError) as error:
        st.error(f"The trained model could not be loaded: {error}")
        return

    upload = st.file_uploader("Upload a fruit photo", type=["jpg", "jpeg", "png", "webp"], help="JPG, PNG, or WebP, up to 10 MB.")
    image_bytes = upload.getvalue() if upload is not None else None

    if image_bytes is None:
        st.divider()
        st.markdown("### One photo. A little more knowledge.")
        st.write("🍎 **Apples** · 🍌 **Bananas**")
        return
    if len(image_bytes) > 10 * 1024 * 1024:
        st.error("Please choose an image smaller than 10 MB.")
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as source:
                photo = ImageOps.exif_transpose(source).copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        st.error("That file could not be read as an image. Try another JPG, PNG, or WebP photo.")
        return
    preview, result = st.columns([1, 1], gap="large")
    with preview:
        st.image(photo, caption="Your fruit photo", use_column_width=True)
    with st.spinner("Taking a look…"):
        fruit, probabilities = predict(model, photo)
    with result:
        st.subheader(f"{'🍎' if fruit == 'apple' else '🍌'} {fruit.title()}")
        if probabilities[fruit] < 0.7:
            st.warning("The prediction is uncertain. Try a clearer photo with the fruit centered.")
    try:
        info = read_fruit_info()
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.warning(f"Fruit information is unavailable: {error}")
    else:
        show_fruit_details(fruit, info[fruit], hashlib.sha256(image_bytes).hexdigest())
    with st.expander("About this classifier"):
        st.write("A custom convolutional neural network trained from random weights on the supplied apple and banana images.")
        metrics_path = ROOT / "artifacts" / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.write(f"Held-out test accuracy: **{metrics['test']['accuracy']:.1%}** across {metrics['test']['count']} images.")
        st.caption("Photos with different backgrounds, lighting, or multiple fruits may produce incorrect predictions.")


if __name__ == "__main__":
    main()
