import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="MM Risk Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
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
    
    .highlight-box {
        background: #e8f4f3;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3d7f7d;
        margin: 1rem 0;
    }
    
    .stat-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="header-container">
    <div class="header-title">🧬 MM Risk Predictor</div>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
        Machine Learning-Based Prognostic Modeling for Multiple Myeloma
    </p>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📌 Introduction</div>', unsafe_allow_html=True)

st.markdown("""
**다발성 골수종(Multiple Myeloma, MM)**은 유전적 이질성이 매우 큰 혈액암으로, 
동일한 초기 치료를 받더라도 전체 생존율(Overall Survival, OS)이 환자마다 크게 다를 수 있습니다.

본 연구는 **MAQC-II 프로젝트**의 다발성 골수종 유전자 발현 데이터를 활용하여 
**ML 기반 예후 예측 모델**을 구축하였습니다.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown("**🎯 연구 목적**")
    st.markdown("""
    - 진단 시점에서 **고위험 환자를 조기 선별**
    - 맞춤형 치료 전략 수립 지원
    - 정밀 의학 기반 예후 예측 도구 개발
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown("**📊 데이터셋**")
    st.markdown("""
    - **Training**: Total Therapy 2 (n=340)
    - **Validation**: Total Therapy 3 (n=214)
    - **Outcome**: 2-year Overall Survival
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Methods
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔬 Methods</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**1️⃣ Feature Selection**")
    st.markdown('<div class="stat-item">', unsafe_allow_html=True)
    st.markdown("""
    - ANOVA 필터링
    - Recursive Feature Elimination (RFE)
    - **20,000개 → 200개 유전자 선정**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("**2️⃣ Model Development**")
    st.markdown('<div class="stat-item">', unsafe_allow_html=True)
    st.markdown("""
    - Logistic Regression
    - Random Forest
    - **XGBoost** (최종 선택)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown("**3️⃣ Validation**")
    st.markdown('<div class="stat-item">', unsafe_allow_html=True)
    st.markdown("""
    - TT3 독립 검증
    - 10분위 위험 계층화
    - 단조적 상관관계 분석
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Key Features
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">⭐ 주요 특징</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: #e8f4f3; border-radius: 10px;">
        <div style="font-size: 3rem; color: #2d5f5d;">200</div>
        <div style="color: #6c757d; font-weight: 600;">선정 유전자</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: #e8f4f3; border-radius: 10px;">
        <div style="font-size: 3rem; color: #2d5f5d;">0.888</div>
        <div style="color: #6c757d; font-weight: 600;">Spearman Rho</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: #e8f4f3; border-radius: 10px;">
        <div style="font-size: 3rem; color: #2d5f5d;">0→100%</div>
        <div style="color: #6c757d; font-weight: 600;">사망률 범위</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Top Contributing Genes
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧬 Top 10 Contributing Genes</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    1. **SPARC** - Known MM biomarker
    2. **C2orf74/KIAA1841**
    3. **FAM105A**
    4. **AKR1C3**
    5. **EPS8L3**
    """)

with col2:
    st.markdown("""
    6. **IL2** - Immune-related
    7. **SNX2**
    8. **LOC100506125**
    9. **CD58** - Known MM biomarker
    10. **ARHGEF37**
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Navigation Guide
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📖 사용 가이드</div>', unsafe_allow_html=True)

st.markdown("""
왼쪽 사이드바에서 페이지를 선택하여 이동할 수 있습니다:

**📊 예측 실행**
- 환자 데이터 CSV 업로드
- 실시간 위험도 예측
- 환자별 위험군 분류

**📈 결과 분석**
- 모델 성능 지표
- 위험도-사망률 상관관계
- Decile별 분포 시각화
""")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6c757d; margin-top: 2rem;">
    <p>Machine Learning-Based Prognostic Modeling for Multiple Myeloma</p>
    <p style="font-size: 0.9rem;">MAQC-II Gene Expression Data | XGBoost Classification Model</p>
</div>
""", unsafe_allow_html=True)
