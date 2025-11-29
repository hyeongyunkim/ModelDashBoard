import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# 0. 기본 설정
# -------------------------------------------------------
st.set_page_config(
    page_title="MM 예후 예측 대시보드",
    layout="wide"
)

# -------------------------------------------------------
# 1. 모델 + feature 리스트 로드
# -------------------------------------------------------
@st.cache_resource
def load_model_and_features():
    model = joblib.load("xgb_mm_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, feature_cols

model, feature_cols = load_model_and_features()

st.title("🧬 Multiple Myeloma 예후 예측 대시보드 (XGBoost)")

st.markdown("""
### 📌 모델 설명  
- 입력: **10개 샘플 유전자**  
- 모델: **최종 XGBoost 생존 예측 모델**  
- 유전자: 최종 선정된 feature 200개  
- 목적: **사망 위험도(0~1)** 점수 + **Very Low ~ Very High 등급 분류**  
""")

# -------------------------------------------------------
# 2. 사용자 입력 구간
# -------------------------------------------------------
st.sidebar.header("📥 입력 데이터 설정")

input_option = st.sidebar.radio(
    "입력 방식 선택",
    ["테스트용 샘플 보기", "CSV 업로드(사용자 입력)"]
)

# CSV 업로드 처리
if input_option == "CSV 업로드(사용자 입력)":
    uploaded = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded is not None:
        user_df = pd.read_csv(uploaded)
        st.success("업로드 성공!")
    else:
        st.warning("CSV 파일을 업로드해주세요.")
        user_df = None  # 업로드 전까지는 None
else:
    st.sidebar.info("샘플 데이터를 사용하려면 example_input.csv가 필요합니다.")
    # user_df = pd.read_csv("example_input.csv")  # 주석 처리
    user_df = None  # 샘플 파일이 없으면 None

# -------------------------------------------------------
# 3. 입력 데이터 확인
# -------------------------------------------------------
st.subheader("📊 입력 데이터 미리보기")

if user_df is not None:
    st.dataframe(user_df.head())
else:
    st.info("데이터를 업로드하거나 선택해주세요.")

# -------------------------------------------------------
# 4. 예측 함수 정의
# -------------------------------------------------------
from sklearn.preprocessing import StandardScaler

def run_prediction(df):
    df = df.copy()

    # 필요한 feature만 사용
    df = df[feature_cols]

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # 위험도 예측
    risk = model.predict_proba(X_scaled)[:, 1]

    # 위험도 구간 나누기
    bins = ["Very Low", "Low", "Medium", "High", "Very High"]
    df_result = pd.DataFrame({
        "Risk_Score": risk,
        "Pred_Group": pd.qcut(risk, 5, labels=bins)
    })
    return df_result

# -------------------------------------------------------
# 5. 예측 실행 버튼
# -------------------------------------------------------
st.subheader("🧪 예측 실행")

if st.button("예측하기"):
    if user_df is None:
        st.error("먼저 데이터를 업로드해주세요!")
    else:
        try:
            result_df = run_prediction(user_df)

            st.success("예측 완료!")
            st.write("### 🩸 예측 결과")
            st.dataframe(result_df)

            # -------------------------------------------------------
            # 6. 시각화 (히스토그램 + 박스플롯)
            # -------------------------------------------------------
            st.markdown("### 📈 Risk Score Distribution")

            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.histplot(result_df["Risk_Score"], bins=20, kde=True, ax=ax1)
            st.pyplot(fig1)

            st.markdown("### 📊 Risk Group Boxplot")

            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.boxplot(x="Pred_Group", y="Risk_Score", data=result_df, ax=ax2)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.info("⚠ 업로드한 CSV가 feature_cols.pkl의 구성과 맞는지 확인하세요.")

else:
    st.info("좌측에서 데이터를 선택하고 **예측하기** 버튼을 눌러주세요.")
