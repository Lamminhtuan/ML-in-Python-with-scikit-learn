import streamlit as st
from PIL import Image
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
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)