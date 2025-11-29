import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -------------------------------------------------------
# 0. 기본 설정
# -------------------------------------------------------
st.set_page_config(
    page_title="MM Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    /* 전체 배경 */
    .main {
        background-color: #f0f2f6;
    }
    
    /* 헤더 스타일 */
    .header-container {
        background: linear-gradient(135deg, #2d5f5d 0%, #3d7f7d 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    /* 카드 스타일 */
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
    
    /* 사이드바 스타일 */
    .sidebar-card {
        background: #e8f4f3;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* 위험도 배지 */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .risk-very-low {
        background-color: #d4edda;
        color: #155724;
    }
    
    .risk-low {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .risk-very-high {
        background-color: #f5c6cb;
        color: #721c24;
    }
    
    /* 테이블 스타일 */
    .patient-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .patient-table th {
        background-color: #f8f9fa;
        padding: 1rem;
        text-align: left;
        font-weight: 600;
        color: #2d5f5d;
    }
    
    .patient-table td {
        padding: 1rem;
        border-bottom: 1px solid #e9ecef;
    }
    
    /* 통계 카드 */
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

# -------------------------------------------------------
# 1. 모델 + feature 리스트 로드
# -------------------------------------------------------
@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load("xgb_mm_model.pkl")
        feature_cols = joblib.load("feature_cols.pkl")
    except FileNotFoundError:
        st.error("⚠️ 모델 파일(xgb_mm_model.pkl, feature_cols.pkl)이 없습니다!")
        st.info("팀원이 만든 모델 파일을 업로드해주세요.")
        return None, None
    return model, feature_cols

model, feature_cols = load_model_and_features()

if model is None or feature_cols is None:
    st.stop()

# -------------------------------------------------------
# 헤더
# -------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title">
        🧬 MM Risk Predictor
    </div>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Multiple Myeloma 예후 예측 시스템</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 2. 사이드바 - 환자 목록
# -------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h2 style="margin: 0; color: #2d5f5d;">환자 목록</h2>
</div>
""", unsafe_allow_html=True)

# 검색 기능
search_query = st.sidebar.text_input("🔍 검색...", "")

# 환자 데이터 업로드
uploaded = st.sidebar.file_uploader("📁 CSV 파일 업로드", type=["csv"])

if uploaded is not None:
    user_df = pd.read_csv(uploaded)
else:
    user_df = None
    st.sidebar.info("CSV 파일을 업로드해주세요.")

# -------------------------------------------------------
# 3. 메인 영역
# -------------------------------------------------------

if user_df is not None:
    # 예측 함수 정의
    from sklearn.preprocessing import StandardScaler

    def run_prediction(df):
        df = df.copy()
        df = df[feature_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        risk = model.predict_proba(X_scaled)[:, 1]
        
        def get_risk_group(score):
            if score < 0.2:
                return "초고위험"
            elif score < 0.4:
                return "고위험"
            elif score < 0.6:
                return "중간위험"
            elif score < 0.8:
                return "저위험"
            else:
                return "초저위험"
        
        df_result = pd.DataFrame({
            "Patient_ID": [f"MM-2025-{str(i+1).zfill(3)}" for i in range(len(risk))],
            "생존율": [f"{int(r*100)}%" for r in risk],
            "위험군": [get_risk_group(r) for r in risk],
            "Risk_Score": risk,
            "최종_업데이트": [datetime.now().strftime("%Y-%m-%d") for _ in range(len(risk))]
        })
        return df_result
    
    # 예측 실행
    result_df = run_prediction(user_df)
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📊 환자 목록", "📈 통계 분석"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">최신 환자 목록</div>', unsafe_allow_html=True)
            
            # 정렬 옵션
            sort_col1, sort_col2, sort_col3 = st.columns([1, 1, 2])
            with sort_col1:
                sort_option = st.selectbox("정렬:", ["최신순", "생존율", "위험군"])
            
            # 테이블 생성
            for idx, row in result_df.head(10).iterrows():
                risk_class = ""
                if "초고위험" in row["위험군"]:
                    risk_class = "risk-very-high"
                elif "고위험" in row["위험군"]:
                    risk_class = "risk-high"
                elif "중간위험" in row["위험군"]:
                    risk_class = "risk-medium"
                elif "저위험" in row["위험군"]:
                    risk_class = "risk-low"
                else:
                    risk_class = "risk-very-low"
                
                st.markdown(f"""
                <div style="background: white; padding: 1rem; margin-bottom: 0.5rem; border-radius: 5px; border-left: 4px solid #2d5f5d;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div><strong>{row['Patient_ID']}</strong></div>
                        <div><strong style="font-size: 1.2rem; color: #2d5f5d;">{row['생존율']}</strong></div>
                        <div><span class="risk-badge {risk_class}">{row['위험군']}</span></div>
                        <div style="color: #6c757d;">{row['최종_업데이트']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # 통계 카드들
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-number">{len(result_df)}명</div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-label">총 환자</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            high_risk = len(result_df[result_df["위험군"].isin(["고위험", "초고위험"])])
            st.markdown('<div class="stat-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-number" style="color: #dc3545;">{high_risk}명</div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-label">고위험군</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            avg_survival = int(result_df["Risk_Score"].mean() * 100)
            st.markdown('<div class="stat-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-number" style="color: #28a745;">{avg_survival}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-label">평균 생존율</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">위험도 분포</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 위험군별 분포 막대그래프
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            risk_counts = result_df["위험군"].value_counts()
            colors = ['#d4edda', '#fff3cd', '#ffc107', '#f8d7da', '#dc3545']
            risk_counts.plot(kind='bar', ax=ax1, color=colors[:len(risk_counts)])
            ax1.set_title('위험군별 환자 수', fontsize=14, fontweight='bold', pad=20)
            ax1.set_xlabel('')
            ax1.set_ylabel('환자 수', fontsize=11)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # 생존율 분포 히스토그램
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.hist(result_df["Risk_Score"] * 100, bins=20, color='#3d7f7d', edgecolor='white')
            ax2.set_title('생존율 분포', fontsize=14, fontweight='bold', pad=20)
            ax2.set_xlabel('생존율 (%)', fontsize=11)
            ax2.set_ylabel('환자 수', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig2)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 위험도 박스플롯
        st.markdown('<div class="card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">위험군별 생존율 상세 분석</div>', unsafe_allow_html=True)
        
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        sns.boxplot(x="위험군", y="Risk_Score", data=result_df, ax=ax3, palette="Set2")
        ax3.set_title('위험군별 생존율 분포', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('위험군', fontsize=11)
        ax3.set_ylabel('생존율', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig3)
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # 데이터가 없을 때
    st.markdown("""
    <div class="card" style="text-align: center; padding: 3rem;">
        <h3 style="color: #6c757d;">📁 CSV 파일을 업로드해주세요</h3>
        <p style="color: #6c757d;">좌측 사이드바에서 환자 데이터 파일을 업로드하면<br>자동으로 예측 결과를 확인할 수 있습니다.</p>
    </div>
    """, unsafe_allow_html=True)
