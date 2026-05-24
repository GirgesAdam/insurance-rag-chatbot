import streamlit as st

st.set_page_config(
    page_title="SecureLife Assist",
    page_icon="🛡️",
    layout="wide",
)

pages = [
    st.Page(
        "main.py",
        title="SecureLife Assist",
        icon="🛡️",
    ),
    st.Page(
        "admin.py",
        title="Admin Portal",
        icon="🔐",
    ),
]

pg = st.navigation(pages)
pg.run()