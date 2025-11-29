import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="MM 예후 예측 대시보드",
    layout="wide"
)

@st.cache_resource
def load_model_and_features():
    model = joblib.load("xgb_mm_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, feature_cols

model, feature_cols = load_model_and_features()

st.title("🧬 Multiple Myeloma 예후 예측 대시보드 (XGBoost)")

st.markdown("""
- 한 줄 = 한 명의 환자  
- 한 컬럼 = 최종 선정된 유전자 200개  
- 값 = 각 유전자의 발현량 (학습 데이터와 스케일 통일)
""")

# -----------------------------
# 1. 데이터 입력 영역 (사이드바)
# -----------------------------
st.sidebar.header("입력 데이터 설정")

input_mode = st.sidebar.radio(
    "입력 방식 선택",
    ["CSV 업로드", "샘플 데이터 사용"],
)

if input_mode == "CSV 업로드":
    uploaded_file = st.sidebar.file_uploader(
        "유전자 발현 CSV 업로드 (.csv)",
        type=["csv"]
    )
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = None
else:
    # 샘플 데이터 사용
    data = pd.read_csv("example_input.csv")
    st.sidebar.info("샘플 데이터(예: 학습 데이터 일부)를 사용 중입니다.")

# -----------------------------
# 2. 예측 실행 버튼
# -----------------------------
run_pred = st.sidebar.button("🔮 예측 실행")

if not run_pred:
    st.info("왼쪽 사이드바에서 데이터를 선택하고 **🔮 예측 실행** 버튼을 눌러줘.")
    st.stop()

if data is None:
    st.error("CSV 파일을 업로드 해주세요.")
    st.stop()

# -----------------------------
# 3. 컬럼 체크 & 정리
# -----------------------------
st.subheader("1️⃣ 입력 데이터 확인")

st.write(f"입력 데이터 shape: `{data.shape[0]} samples × {data.shape[1]} columns`")

missing_cols = [c for c in feature_cols if c not in data.columns]
extra_cols = [c for c in data.columns if c not in feature_cols]

if missing_cols:
    st.error(f"🌋 필수 유전자 {len(missing_cols)}개가 빠져 있습니다.\n\n예시: {missing_cols[:10]}")
    st.stop()

if extra_cols:
    st.warning(f"참고: 모델에서 사용하지 않는 컬럼 {len(extra_cols)}개가 있습니다. (무시됨)\n\n예시: {extra_cols[:10]}")

# 모델용 X만 추출
X = data[feature_cols]

st.write("입력 데이터 미리보기 (상위 5행)")
st.dataframe(X.head())

# -----------------------------
# 4. 예측 수행
# -----------------------------
st.subheader("2️⃣ 예측 결과 계산")

# XGBoost: class 1(사망)의 확률을 Risk Score로 사용
probas = model.predict_proba(X)[:, 1]
data_result = data.copy()
data_result["Risk_Score"] = probas

# Risk Score 기반 quantile 그룹 나누기 (Very Low ~ Very High)
n_bins = 5
try:
    bins = np.quantile(probas, [0, 0.2, 0.4, 0.6, 0.8, 1])
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    data_result["Risk_Group"] = pd.cut(
        probas,
        bins=bins,
        labels=labels,
        include_lowest=True,
        duplicates="drop"
    )
except Exception as e:
    # 혹시 quantile이 겹치면 equal-width로 대체
    st.warning(f"Quantile 분리가 실패해서 equal-width로 대체했습니다. ({e})")
    bins = n_bins
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    data_result["Risk_Group"] = pd.cut(
        probas,
        bins=bins,
        labels=labels
    )

st.success("✅ 예측 완료!")

# 핵심 결과 테이블
st.write("### 예측 결과 테이블 (앞 10명만 표시)")
st.dataframe(
    data_result[["Risk_Score", "Risk_Group"]].head(10).style.format(
        {"Risk_Score": "{:.4f}"}
    )
)

# -----------------------------
# 5. 시각화: Risk Score 분포
# -----------------------------
st.subheader("3️⃣ Risk Score 분포 시각화")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 전체 Risk Score 분포 (Histogram)")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(data_result["Risk_Score"], kde=True, ax=ax)
    ax.set_xlabel("Risk Score (High → Death Likelihood)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3, axis="y")
    st.pyplot(fig)

with col2:
    st.markdown("#### 🎯 Risk Group별 Score 분포 (Boxplot)")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    order = ["Very Low", "Low", "Medium", "High", "Very High"]
    sns.boxplot(
        x="Risk_Group",
        y="Risk_Score",
        data=data_result,
        order=order,
        ax=ax2
    )
    ax2.set_xlabel("Predicted Risk Group")
    ax2.set_ylabel("Risk Score")
    ax2.grid(alpha=0.3, axis="y")
    st.pyplot(fig2)

# -----------------------------
# 6. 그룹별 요약 통계
# -----------------------------
st.subheader("4️⃣ Risk Group별 요약 통계")

group_summary = (
    data_result
    .groupby("Risk_Group")["Risk_Score"]
    .agg(["count", "mean", "min", "max"])
    .reindex(["Very Low", "Low", "Medium", "High", "Very High"])
)

st.dataframe(group_summary.style.format("{:.4f}"))
