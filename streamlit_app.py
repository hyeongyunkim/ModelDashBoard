import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# 페이지 설정
# -------------------------------------------------------
st.set_page_config(
    page_title="MM Risk Predictor",
    page_icon="🧬",
    layout="wide"
)

# -------------------------------------------------------
# 커스텀 CSS
# -------------------------------------------------------
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
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    .card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2d5f5d;
        margin-bottom: 1rem;
        border-left: 4px solid #3d7f7d;
        padding-left: 1rem;
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
    
    .info-box {
        background: #e8f4f3;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3d7f7d;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 모델 로드
# -------------------------------------------------------
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

# -------------------------------------------------------
# 헤더
# -------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title">🧬 MM Risk Predictor</div>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
        XGBoost-based Multiple Myeloma Prognosis Prediction
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 탭 생성
# -------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Predict My Sample", "📋 Clinical Interpretation"])

# =======================================================
# 탭 1: Predict My Sample
# =======================================================
with tab1:
    st.markdown('<div class="section-title">📁 Upload Patient Data</div>', unsafe_allow_html=True)
    
    # CSV 업로드
    uploaded = st.file_uploader("Upload CSV file with gene expression data", type=["csv"])
    
    if uploaded is None:
        # 업로드 안내
        st.markdown("""
        <div class="upload-container">
            <div style="font-size: 4rem; color: #3d7f7d; margin-bottom: 1rem;">📁</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #2d5f5d; margin-bottom: 0.5rem;">
                Upload Gene Expression Data
            </div>
            <div style="font-size: 1rem; color: #6c757d;">
                CSV file with 200 gene features required
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("📋 **Required format**: CSV file with 200 gene expression features matching the model's feature set")
    
    else:
        # 파일 업로드됨
        try:
            user_df = pd.read_csv(uploaded)
            
            # Feature 일치 여부 자동 검사
            st.markdown('<div class="section-title">✅ Data Validation</div>', unsafe_allow_html=True)
            
            missing_features = set(feature_cols) - set(user_df.columns)
            extra_features = set(user_df.columns) - set(feature_cols)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Uploaded Samples", len(user_df))
            with col2:
                st.metric("Required Features", len(feature_cols))
            with col3:
                st.metric("Matched Features", len(set(feature_cols) & set(user_df.columns)))
            
            if missing_features:
                st.error(f"❌ Missing {len(missing_features)} required features")
                with st.expander("Show missing features"):
                    st.write(list(missing_features)[:10])
                st.stop()
            
            if extra_features:
                st.warning(f"⚠️ Found {len(extra_features)} extra columns (will be ignored)")
            
            st.success("✅ All required features found! Ready for prediction.")
            
            # -------------------------------------------------------
            # 예측 함수
            # -------------------------------------------------------
            def run_prediction(df):
                df = df.copy()
                df = df[feature_cols]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df)
                
                # Risk Score 계산
                risk = model.predict_proba(X_scaled)[:, 1]
                
                # Risk Group 분류
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
                    "Patient_ID": [f"MM-{str(i+1).zfill(3)}" for i in range(len(risk))],
                    "Risk_Score": risk,
                    "Risk_Group": [get_risk_group(r) for r in risk],
                })
                return df_result
            
            # 예측 실행
            st.markdown('<div class="section-title">🔬 Prediction Results</div>', unsafe_allow_html=True)
            
            result_df = run_prediction(user_df)
            
            # 통계 요약
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
                st.markdown('<div class="stat-label">High Risk</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                medium_risk = len(result_df[result_df["Risk_Group"] == "Medium Risk"])
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-number" style="color: #ffc107;">{medium_risk}</div>', unsafe_allow_html=True)
                st.markdown('<div class="stat-label">Medium Risk</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                low_risk = len(result_df[result_df["Risk_Group"].isin(["Low Risk", "Very Low Risk"])])
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-number" style="color: #28a745;">{low_risk}</div>', unsafe_allow_html=True)
                st.markdown('<div class="stat-label">Low Risk</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 결과 테이블
            st.markdown("### 📋 Patient-wise Results")
            
            # Risk Group별 색상 매핑
            def color_risk_group(val):
                colors = {
                    "Very High Risk": "background-color: #f5c6cb",
                    "High Risk": "background-color: #f8d7da",
                    "Medium Risk": "background-color: #fff3cd",
                    "Low Risk": "background-color: #d1ecf1",
                    "Very Low Risk": "background-color: #d4edda"
                }
                return colors.get(val, "")
            
            styled_df = result_df.style.applymap(color_risk_group, subset=['Risk_Group'])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # 시각화
            st.markdown("### 📊 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk Score Histogram
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                ax1.hist(result_df["Risk_Score"], bins=20, color='#3d7f7d', edgecolor='white', alpha=0.7)
                ax1.axvline(result_df["Risk_Score"].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {result_df["Risk_Score"].mean():.3f}')
                ax1.set_xlabel('Risk Score (Death Probability)', fontsize=11, fontweight='bold')
                ax1.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
                ax1.set_title('Risk Score Distribution', fontsize=13, fontweight='bold', pad=15)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig1)
            
            with col2:
                # Risk Group Boxplot
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                
                # Risk Group 순서 정의
                risk_order = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
                result_df['Risk_Group'] = pd.Categorical(result_df['Risk_Group'], categories=risk_order, ordered=True)
                
                sns.boxplot(x="Risk_Group", y="Risk_Score", data=result_df, ax=ax2, palette="RdYlGn_r")
                ax2.set_xlabel('Risk Group', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Risk Score', fontsize=11, fontweight='bold')
                ax2.set_title('Risk Score by Risk Group', fontsize=13, fontweight='bold', pad=15)
                plt.xticks(rotation=45, ha='right')
                ax2.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig2)
            
            # 다운로드 버튼
            st.markdown("### 💾 Download Results")
            
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Prediction Results (CSV)",
                data=csv,
                file_name=f"MM_Risk_Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check your CSV file format and try again.")

# =======================================================
# 탭 2: Clinical Interpretation
# =======================================================
with tab2:
    st.markdown('<div class="section-title">📋 Understanding Your Results</div>', unsafe_allow_html=True)
    
    # Risk Score 설명
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 What is Risk Score?")
    st.markdown("""
    **Risk Score**는 환자의 **2년 내 사망 확률**을 나타냅니다.
    
    - **0에 가까울수록**: 낮은 사망 위험 (높은 생존율)
    - **1에 가까울수록**: 높은 사망 위험 (낮은 생존율)
    
    이 점수는 200개의 핵심 유전자 발현 패턴을 XGBoost 모델이 분석하여 계산됩니다.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk Group 설명
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏥 Risk Group Classification")
    st.markdown("""
    환자는 Risk Score를 기반으로 **5개의 위험군**으로 분류됩니다:
    """)
    
    risk_groups = pd.DataFrame({
        "Risk Group": ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
        "Risk Score Range": ["0.8 - 1.0", "0.6 - 0.8", "0.4 - 0.6", "0.2 - 0.4", "0.0 - 0.2"],
        "Clinical Meaning": [
            "매우 낮은 사망 위험, 표준 치료 권장",
            "낮은 사망 위험, 정기 모니터링",
            "중간 사망 위험, 집중 관찰 필요",
            "높은 사망 위험, 적극적 치료 고려",
            "매우 높은 사망 위험, 강화 치료 필수"
        ]
    })
    
    st.dataframe(risk_groups, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Decile 분석
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Decile Analysis Summary")
    st.markdown("""
    본 모델은 **독립 검증 데이터셋(TT3, n=214)**에서 뛰어난 성능을 입증했습니다.
    
    환자를 위험도 기준 10분위로 나눈 결과:
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Decile 사망률 그래프
        decile_data = pd.DataFrame({
            'Decile': list(range(1, 11)),
            'Mortality_Rate': [0, 10, 20, 30, 45, 60, 72, 85, 93, 100]
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(decile_data['Decile'], decile_data['Mortality_Rate'], 
                marker='o', linewidth=3, markersize=12, color='#dc3545')
        ax.fill_between(decile_data['Decile'], decile_data['Mortality_Rate'], 
                         alpha=0.2, color='#dc3545')
        ax.set_xlabel('Risk Decile (1=Lowest Risk, 10=Highest Risk)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mortality Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Mortality Rate by Risk Decile (Validation Cohort)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(-5, 105)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        #### 주요 발견
        
        **Spearman's Rho = 0.888**  
        (p < 0.001)
        
        - 1분위: 사망률 **0%**
        - 10분위: 사망률 **100%**
        
        ➡️ 예측 위험도와 실제 사망률 간 **강한 단조적 상관관계** 확인
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Top 10 유전자
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧬 Top 10 Contributing Genes")
    st.markdown("""
    모델의 예측에 가장 크게 기여하는 **10개 유전자**:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        1. **SPARC** - 세포외 기질 단백질, MM 바이오마커
        2. **C2orf74/KIAA1841** - 염색체 2번 유전자
        3. **FAM105A** - 세포 기능 조절
        4. **AKR1C3** - 스테로이드 대사
        5. **EPS8L3** - 세포 신호 전달
        """)
    
    with col2:
        st.markdown("""
        6. **IL2** - 면역 반응 조절
        7. **SNX2** - 세포 내 수송
        8. **LOC100506125** - 기능 미확인
        9. **CD58** - 면역 조절, MM 마커
        10. **ARHGEF37** - Rho GTPase 조절
        """)
    
    st.info("💡 **SPARC**와 **CD58**은 Multiple Myeloma에서 이미 잘 알려진 바이오마커로, 모델의 생물학적 타당성을 뒷받침합니다.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 고위험군의 중요성
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚠️ Why High-Risk Patients Matter")
    st.markdown("""
    **고위험 환자 조기 식별**은 다발성 골수종 치료에서 매우 중요합니다:
    
    1. **치료 강도 결정**
       - 고위험 환자 → 더 적극적인 초기 치료
       - 저위험 환자 → 부작용 최소화한 표준 치료
    
    2. **임상시험 참여**
       - 고위험군을 대상으로 한 신약 임상시험
       - 맞춤형 치료법 개발
    
    3. **모니터링 주기**
       - 고위험: 집중 추적 관찰
       - 저위험: 정기 검진
    
    4. **예후 상담**
       - 환자와 가족에게 정확한 예후 정보 제공
       - 치료 계획 수립 지원
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 임상 활용
    st.markdown('<div class="card" style="background: #e8f4f3; border-left: 4px solid #2d5f5d;">', unsafe_allow_html=True)
    st.markdown("### 💡 Clinical Applications")
    st.markdown("""
    이 예측 모델은 다음과 같이 임상에서 활용될 수 있습니다:
    
    ✅ **진단 시점 위험 평가**
    - 새로 진단된 MM 환자의 예후 예측
    - 치료 전략 수립의 객관적 근거 제공
    
    ✅ **개인 맞춤형 치료**
    - 위험군별 차별화된 치료 프로토콜
    - 불필요한 과치료/과소치료 방지
    
    ✅ **임상 의사결정 지원**
    - 200개 유전자 발현 데이터 기반
    - 객관적이고 재현 가능한 예측
    
    ✅ **정밀 종양학 실현**
    - 분자 수준의 환자 계층화
    - 치료 성과 개선 가능성
    
    ---
    
    **⚠️ 중요**: 이 도구는 임상 의사결정을 **보조**하는 목적으로 개발되었으며, 
    최종 치료 결정은 반드시 전문의의 종합적인 판단 하에 이루어져야 합니다.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
