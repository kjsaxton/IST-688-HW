import streamlit as st

st.set_page_config(page_title="HW Manager")

HW1 = st.Page('HW/HW1.py', title='HW1')
HW2 = st.Page('HW/HW2.py', title='HW2')
HW3 = st.Page('HW/HW3.py', title='HW3', default=True)

pg = st.navigation([HW1, HW2, HW3])
pg.run()
