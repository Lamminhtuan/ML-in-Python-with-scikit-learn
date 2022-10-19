import streamlit as st
from PIL import Image
import cv2
import numpy as np
st.markdown("""
    # Đây là bài tutorial
    ## 1. Giới thiệu streamlit
    ### 1.1. Giới thiệu chung
    ### 1.2. Cài đặt
    ## 2. Các thành phần cơ bản của giao diện

    """
)
cat_tab, dog_tab, owl_tab = st.tabs(["Cat", "Dog", "Owl"])
with cat_tab:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")
with dog_tab:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg")
with owl_tab:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")
a_value = st.text_input("Nhập a:")
b_value = st.text_input("Nhập b:")
operator = st.radio("Chọn phép toán",['Cộng', 'Trừ', 'Nhân', 'Chia'])
button = st.button('Tính')
if button:
    if operator == 'Cộng':
        st.text_input("Kết quả:",float(a_value) + float(b_value))
    if operator == 'Trừ':
        st.text_input("Kết quả:",float(a_value) - float(b_value))
    if operator == 'Nhân':
        st.text_input("Kết quả:",float(a_value) * float(b_value))
    if operator =='Chia':
        st.text_input("Kết quả:",float(a_value) / float(b_value))
uploaded_file = st.file_uploader("Chọn ảnh")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)
    
    with open(uploaded_file.name, 'wb') as f:
        f.write(bytes_data)
    img = cv2.imread(uploaded_file.name)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    filer = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])
    result = cv2.filter2D(img, -1, filter)
    st.image(Image.fromarray(img))