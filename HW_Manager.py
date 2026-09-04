import streamlit as st

HW1=st.Page('HW1.py', title='HW1')
HW2=st.Page('HW2.py', title='HW2', default=True)

pg=st.navigation([Lab1,Lab2])
pg.run()
