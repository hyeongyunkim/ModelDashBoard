import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# 페이지 설정
st.set_page_config(
    page_title="예측 실행 - MM Risk Predictor",
    page_icon="📊",
    layout="wide"
)

# 커스텀 CSS (이전과 동일)
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    
    .header-container {
        background: linear-gradient(135deg, #2d5f5d 0%, #3d7f7d 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2d5f5d;
    }
    
    .upload-container {
        background: white;
        padding: 3rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px dashed #3d7f7d;
        margin: 2rem 0;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .risk-very-low { background-color: #d4edda; color: #155724; }
    .risk-low { background-color: #d1ecf1; color: #0c5460; }
    .risk-medium { background-color: #fff3cd; color: #856404; }
    .risk-high { background-color: #f8d7da; color: #721c24; }
    .risk-very-high { background-color: #f5c6cb; color: #721c24; }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #2d5f5d;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 모델 로드
@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load("xgb_mm_model.pkl")
        feature_cols = joblib.load("feature_cols.pkl")
    except FileNotFoundError:
        st.error("⚠️ 모델 파일(xgb_mm_model.pkl, feature_cols.pkl)이 없습니다!")
        return None, None
    return model, feature_cols

model, feature_cols = load_model_and_features()

if model is None or feature_cols is None:
    st.stop()

# 헤더
st.markdown("""
<div class="header-container">
    <div style="font-size: 2.5rem; font-weight: bold;">📊 예측 실행</div>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">환자 데이터 업로드 및 위험도 예측</p>
</div>
""", unsafe_allow_html=True)

# 파일 업로드
uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

if uploaded is None:
    # 업로드 안내 화면
    st.markdown("""
    <div class="upload-container">
        <div style="font-size: 4rem; color: #3d7f7d; margin-bottom: 1rem;">📁</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #2d5f5d; margin-bottom: 0.5rem;">
            환자 유전자 발현 데이터 업로드
        </div>
        <div style="font-size: 1rem; color: #6c757d; margin-bottom: 2rem;">
            200개 유전자 feature가 포함된 CSV 파일을 업로드하세요
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 데이터 형식 안내
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 데이터 형식 요구사항</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **필수 요소:**
        - CSV 파일 형식
        - 200개의 유전자 feature 컬럼
        - 각 행은 하나의 환자 샘플
        """)
    
    with col2:
        st.markdown("""
        **예시 구조:**
```
        GENE_1, GENE_2, ..., GENE_200
        10.5,   8.3,    ..., 12.1
        9.7,    11.2,   ..., 8.9
```
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # 파일 업로드됨
    user_df = pd.read_csv(uploaded)
    
    st.success(f"✅ 파일 업로드 완료! ({len(user_df)}개 샘플)")
    
    # 예측 함수
    def run_prediction(df):
        df = df.copy()
        df = df[feature_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        risk = model.predict_proba(X_scaled)[:, 1]
        
        def get_risk_group(score):
            if score < 0.2:
                return "Very High Risk"
            elif score < 0.4:
                return "High Risk"
            elif score < 0.6:
                return "Medium Risk"
            elif score < 0.8:
                return "Low Risk"
            else:
                return "Very Low Risk"
        
        df_result = pd.DataFrame({
            "Patient_ID": [f"MM-2025-{str(i+1).zfill(3)}" for i in range(len(risk))],
            "Survival_Rate": [f"{int(r*100)}%" for r in risk],
            "Risk_Group": [get_risk_group(r) for r in risk],
            "Risk_Score": risk,
            "Updated": [datetime.now().strftime("%Y-%m-%d") for _ in range(len(risk))]
        })
        return df_result
    
    # 예측 실행
    result_df = run_prediction(user_df)
    
    # 통계 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-number">{len(result_df)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Total Patients</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        high_risk = len(result_df[result_df["Risk_Group"].isin(["High Risk", "Very High Risk"])])
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-number" style="color: #dc3545;">{high_risk}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">High Risk Patients</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        avg_survival = int(result_df["Risk_Score"].mean() * 100)
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-number" style="color: #28a745;">{avg_survival}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Avg. Survival Rate</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        low_risk = len(result_df[result_df["Risk_Group"].isin(["Low Risk", "Very Low Risk"])])
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-number" style="color: #17a2b8;">{low_risk}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Low Risk Patients</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭
    tab1, tab2 = st.tabs(["📋 Patient List", "📊 Visualization"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">환자별 예측 결과</div>', unsafe_allow_html=True)
        
        # 정렬 옵션
        sort_option = st.selectbox("Sort by:", ["Latest", "Survival Rate (High→Low)", "Survival Rate (Low→High)", "Risk Group"])
        
        if sort_option == "Survival Rate (High→Low)":
            display_df = result_df.sort_values("Risk_Score", ascending=False)
        elif sort_option == "Survival Rate (Low→High)":
            display_df = result_df.sort_values("Risk_Score", ascending=True)
        elif sort_option == "Risk Group":
            display_df = result_df.sort_values("Risk_Group")
        else:
            display_df = result_df
        
        # 환자 카드
        for idx, row in display_df.head(20).iterrows():
            risk_class = {
                "Very High Risk": "risk-very-high",
                "High Risk": "risk-high",
                "Medium Risk": "risk-medium",
                "Low Risk": "risk-low",
                "Very Low Risk": "risk-very-low"
            }.get(row["Risk_Group"], "risk-medium")
            
            st.markdown(f"""
            <div style="background: white; padding: 1rem; margin-bottom: 0.5rem; border-radius: 5px; border-left: 4px solid #2d5f5d;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;"><strong>{row['Patient_ID']}</strong></div>
                    <div style="flex: 1; text-align: center;"><strong style="font-size: 1.2rem; color: #2d5f5d;">{row['Survival_Rate']}</strong></div>
                    <div style="flex: 1; text-align: center;"><span class="risk-badge {risk_class}">{row['Risk_Group']}</span></div>
                    <div style="flex: 1; text-align: right; color: #6c757d;">{row['Updated']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 위험군 분포
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            risk_counts = result_df["Risk_Group"].value_counts()
            colors = ['#28a745', '#17a2b8', '#ffc107', '#fd7e14', '#dc3545']
            risk_counts.plot(kind='bar', ax=ax1, color=colors[:len(risk_counts)])
            ax1.set_title('Risk Group Distribution', fontsize=14, fontweight='bold', pad=20)
            ax1.set_xlabel('')
            ax1.set_ylabel('Number of Patients', fontsize=11)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # 생존율 분포
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.hist(result_df["Risk_Score"] * 100, bins=20, color='#3d7f7d', edgecolor='white')
            ax2.set_title('Survival Rate Distribution', fontsize=14, fontweight='bold', pad=20)
            ax2.set_xlabel('Survival Rate (%)', fontsize=11)
            ax2.set_ylabel('Number of Patients', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig2)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 다운로드 버튼
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction Results (CSV)",
        data=csv,
        file_name=f"MM_prediction_results_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
