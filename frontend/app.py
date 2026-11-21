"""Minimal Streamlit UI for Odwołania2json."""

import streamlit as st

st.set_page_config(page_title="Odwołania2json", layout="wide")
st.title("Odwołania2json")
st.caption("Ekstrakcja danych ze spraw odwoławczych (PDF → JSON UZP)")

uploaded_file = st.file_uploader("Wgraj plik PDF ze sprawą odwoławczą", type=["pdf"])

if uploaded_file is not None:
    st.info(f"Wgrano plik: {uploaded_file.name} ({uploaded_file.size} B)")
    # TODO: Wyślij plik do backendu FastAPI (/api/extract) i wyświetl wygenerowany JSON
    st.json({"filename": uploaded_file.name, "status": "stub"})
