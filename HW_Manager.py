import streamlit as st

st.set_page_config(page_title="HW Manager")

HW1 = st.Page('HW/HW1.py', title='HW1')
HW2 = st.Page('HW/HW2.py', title='HW2', default=True)

pg = st.navigation([HW1, HW2])
pg.run()
