import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import pickle


# Import model
nb = pickle.load(open('nb.pkl', 'rb'))

# Load dataset
data = pd.read_csv('Bank_Customer.csv')

st.markdown(
    """
    <style>
        .st-emotion-cache-vk3wp9 {
            background: aliceblue;
        }
        .st-emotion-cache-uf99v8 {
            background: orange;
        }
    </style>
    """,
    unsafe_allow_html=True
)

html_layout1 = """
<div style="display: flex; align-items: center;">
        <img src="https://i.ibb.co/k34LwdF/1.png" alt="" style="width: 100px;">
        <h1>Aplikasi Pindah Bank</h1>
</div>
<br>
<div style="background-color:rgb(0 162 201) ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Pindah Bank Prediksi</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1, unsafe_allow_html=True)
activities = ['NB', 'Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?', activities)
st.sidebar.header('Data Customer')

if st.checkbox("Tentang Dataset"):
    html_layout2 = """
    <br>
    <p>Ini adalah dataset Bank Customer Churn</p>
    """
    st.markdown(html_layout2, unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDA'):
    pr = ProfileReport(data, explorative=True)
    st.header('*Input Dataframe*')
    st.write(data)
    st.write('---')
    st.header('*Profiling Report*')
    st_profile_report(pr)

# train test split
X_train, X_test, y_train, y_test = train_test_split(data.drop('churn', axis=1), data['churn'], test_size=0.20, random_state=42)

# Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    skor_kredit = st.sidebar.slider('Skor Kredit', 350, 850, 850)
    negara = st.sidebar.slider('Negara', 1, 3, 2)
    jk = st.sidebar.slider('Jenis Kelamin', 0, 1, 0)
    usia = st.sidebar.slider('Usia', 18, 92, 43)
    tenure = st.sidebar.slider('Tenure', 1, 10, 2)
    balance = st.sidebar.slider('Balance', 3768.69, 250898.09, 125510.82)
    nomor_produk = st.sidebar.slider('Nomor Produk', 1, 4, 1)
    kartu_kredit = st.sidebar.slider('Kartu Kredit', 0, 1, 1)
    member_aktif = st.sidebar.slider('Member Aktif', 0, 1, 1)
    penghasilan = st.sidebar.slider('Perkiraan Penghasilan', 11.58, 199992.48, 79084.10)

    user_report_data = {
        'credit_score': skor_kredit,
        'country': negara,
        'gender': jk,
        'age': usia,
        'tenure': tenure,
        'balance': balance,
        'products_number': nomor_produk,
        'credit_card': kartu_kredit,
        'active_member': member_aktif,
        'estimated_salary': penghasilan
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Data Bank Customer
user_data = user_report()
st.subheader('Data Bank Churn')
st.write(user_data)


# Make predictions
try:
    user_result = nb.predict(user_data)
    nb_score = accuracy_score(y_test, nb.predict(X_test))

    # Output
    st.subheader('Hasilnya adalah : ')
    output = 'Tetap' if user_result[0] == 0 else 'Berpindah'
    st.title(output)
    st.subheader('Model yang digunakan : \n' + option)
    st.subheader('Accuracy : ')
    st.write(str(nb_score * 100) + '%')
except Exception as e:
    st.error(f"Error during prediction: {e}")
