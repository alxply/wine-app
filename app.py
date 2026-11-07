import json
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import metrics

MODEL_PATH = Path(__file__).parent / "model.joblib"
CONFIG_PATH = Path(__file__).parent / "config.json"
TEST_PATH = Path(__file__).parent / "test.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config.get('threshold', 0.5)

@st.cache_data
def load_test():
    return pd.read_csv(TEST_PATH)

@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]

try:
    model = load_model()
    default_threshold = load_config()
    df_test = load_test()
    X_test = df_test.drop(columns=['target'])
    y_test = df_test['target']
    y_prob = test_proba(model, X_test)
except Exception as e:
    st.error(e)
    st.stop()

st.sidebar.header('Настройка модели')
threshold = st.sidebar.slider(
    'Порог классификации',
    min_value=0.0,
    max_value=1.0,
    value=float(default_threshold),
    step=0.01
)

tab1, tab2 = st.tabs(['Оценка одного вина', 'Метрики'])

with tab1:
    st.header('Оценка одного вина')
    feature_columns = [column for column in df_test.columns if column != 'target']
    input_data = {}
    with st.form('wine_features'):
        for column in feature_columns:
            if column == 'is_red':
                input_data[column] = st.radio(f"{column}", options=[0, 1], format_func=lambda x: 'Красное вино' if x == 1 else 'Белое вино', index=1 if df_test['is_red'].median() == 1 else 0)
            else:
                min_value, max_value, value = float(df_test[column].min()), float(df_test[column].max()), float(df_test[column].median())
                if min_value == max_value:
                    max_value += 1.0
                input_data[column] = st.slider(f"{column}", min_value=min_value, max_value=max_value, value=value, step=0.1)
        submit_button = st.form_submit_button('Оценить вино')
    if submit_button:
        prob = model.predict_proba(pd.DataFrame([input_data])[feature_columns])[0, 1]
        st.metric(label='Вероятность, что вино хорошее', value=f"{prob * 100:.2f}%")
        if prob >= threshold:
            st.success('Хорошее вино')
        else:
            st.error('Плохое вино')

with tab2:
    st.header('Метрики')
    y_pred = (y_prob >= threshold).astype(int)
    cm = metrics.confusion_matrix(y_test, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    disp.plot(cmap='Blues')
    plt.title('Матрица ошибок')
    st.pyplot(disp.figure_)
    plt.close()
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.fbeta_score(y_test, y_pred, beta=1, zero_division=0)
    st.metric(label="Precision", value=f"{precision:.4f}")
    st.metric(label="Recall", value=f"{recall:.4f}")
    st.metric(label="F1", value=f"{f1:.4f}")
