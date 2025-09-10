import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_model():
    model = joblib.load('stacking_classifier_subtype.joblib')
    encoders = joblib.load('classification_encoders.joblib')
    target_encoder = joblib.load('target_encoder_subtype.joblib')
    feature_names = joblib.load('classification_features.joblib')
    return model, encoders, target_encoder, feature_names

try:
    model, encoders, target_encoder, feature_names = load_model()
    st.success("succ")
except Exception as e:
    st.error(f"err {e}")
    st.stop()

reverse_target_map = {i: label for i, label in enumerate(target_encoder.classes_)}

input_data = {} # признаки

st.header("характеристики недвижимости")

for feature in feature_names:
    if feature in encoders:
        le = encoders[feature]
        categories = le.classes_.tolist()
        
        if 'Unknown' not in categories:
            categories.append('Unknown')
        
        selected = st.selectbox(f"{feature}:", categories, key=feature)
        input_data[feature] = selected
    else:
        if feature in ['size', 'price', 'floor_no', 'total_floor_count', 'building_age']:
            val = st.number_input(f"{feature}:", value=0, step=1, key=feature)
        else:
            val = st.text_input(f"{feature} (введите значение):", key=feature)
        input_data[feature] = val

if st.button("predict"):
    try:

        df_input = pd.DataFrame([input_data])
    
        for col, le in encoders.items():
            if col in df_input.columns:

                known_labels = set(le.classes_)
                df_input[col] = df_input[col].apply(lambda x: x if x in known_labels else 'Unknown')
                df_input[col] = le.transform(df_input[col])
        

        df_input = df_input[feature_names]
        
        # проба
        pred_code = model.predict(df_input)[0]
        pred_label = reverse_target_map.get(pred_code, f"Класс {pred_code}")
        

        st.subheader("результат:")
        st.success(f"предсказанный тип недвижимости: **{pred_label}**")
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_input)[0]
            st.write("**Вероятности по классам:**")
            proba_df = pd.DataFrame({
                'Класс': [reverse_target_map.get(i, f"Класс {i}") for i in range(len(proba))],
                'Вероятность': proba
            }).sort_values('Вероятность', ascending=False)
            st.dataframe(proba_df.style.format({'Вероятность': '{:.2%}'}))
    
    except Exception as e:
        st.error(f"err: {e}")
        st.exception(e)
