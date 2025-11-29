import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 페이지 설정
st.set_page_config(
    page_title="결과 분석 - MM Risk Predictor",
    page_icon="📈",
    layout="wide"
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
    
    .metric-box {
        background: #e8f4f3;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2d5f5d;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="header-container">
    <div style="font-size: 2.5rem; font-weight: bold;">📈 결과 분석</div>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">모델 성능 및 임상적 검증 결과</p>
</div>
""", unsafe_allow_html=True)

# Model Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🎯 Model Performance Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

metrics = [
    ("AUC", "0.92"),
    ("MCC", "0.85"),
    ("Recall", "0.89"),
    ("Precision", "0.91"),
    ("F1-Score", "0.90")
]

for col, (label, value) in zip([col1, col2, col3, col4, col5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
최적화된 XGBoost 모델은 모든 평가 지표에서 우수한 성능을 보였으며, 
특히 **MCC(Matthews Correlation Coefficient) 0.85**로 불균형 데이터에서도 강건한 예측력을 입증했습니다.
""")

st.markdown('</div>', unsafe_allow_html=True)

# Risk Stratification
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Risk Stratification Analysis</div>', unsafe_allow_html=True)

# Decile 데이터 (예시)
decile_data = pd.DataFrame({
    'Decile': list(range(1, 11)),
    'Mortality_Rate': [0, 10, 20, 30, 45, 60, 72, 85, 93, 100]
})

col1, col2 = st.columns([2, 1])

with col1:
    # Decile별 사망률 그래프
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(decile_data['Decile'], decile_data['Mortality_Rate'], 
            marker='o', linewidth=3, markersize=10, color='#dc3545')
    ax.fill_between(decile_data['Decile'], decile_data['Mortality_Rate'], 
                     alpha=0.3, color='#dc3545')
    ax.set_xlabel('Risk Decile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mortality Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mortality Rate by Risk Decile', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 11))
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("""
    ### 주요 발견사항
    
    **Spearman's Rho = 0.888**  
    (p = 6.08 × 10⁻⁴)
    
    - 예측 위험도와 실제 사망률 간 **강한 단조적 상관관계** 확인
    - 1분위(최저위험): **사망률 0%**
    - 10분위(최고위험): **사망률 100%**
    
    → 모델의 임상적 타당성 입증
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Top Contributing Genes
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧬 Top 10 Contributing Genes</div>', unsafe_allow_html=True)

# 유전자 중요도 데이터 (예시)
gene_importance = pd.DataFrame({
    'Gene': ['SPARC', 'C2orf74', 'FAM105A', 'AKR1C3', 'EPS8L3', 
             'IL2', 'SNX2', 'LOC100506125', 'CD58', 'ARHGEF37'],
    'Importance': [0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05]
})

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(gene_importance['Gene'], gene_importance['Importance'], color='#3d7f7d')
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Contributing Genes', fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    # 값 표시
    for i, (gene, imp) in enumerate(zip(gene_importance['Gene'], gene_importance['Importance'])):
        ax.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("""
    ### Known Biomarkers
    
    **SPARC**
    - MM에서 잘 알려진 바이오마커
    - 세포외 기질 단백질
    
    **CD58**
    - 면역 조절 관련
    - MM 예후 관련 마커
    
    **IL2**
    - 면역 반응 관련
    - T세포 활성화
    
    → 생물학적 타당성 확보
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Clinical Implications
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💡 Clinical Implications</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ 임상적 활용 가능성
    
    1. **조기 위험 계층화**
       - 진단 시점에서 고위험 환자 식별
       - 치료 강도 결정에 활용
    
    2. **맞춤형 치료 전략**
       - 위험군별 차별화된 치료 프로토콜
       - 임상시험 참여 대상 선정
    
    3. **예후 모니터링**
       - 치료 반응 예측
       - 추적 관찰 주기 결정
    """)

with col2:
    st.markdown("""
    ### 🔬 향후 연구 방향
    
    1. **다기관 검증**
       - 외부 코호트 추가 검증
       - 일반화 성능 평가
    
    2. **다중 오믹스 통합**
       - 임상 지표 통합
       - 세포유전학 데이터 결합
    
    3. **전향적 연구**
       - 실제 임상 영향 평가
       - 규제 승인 준비
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Conclusion
st.markdown('<div class="card" style="background: #e8f4f3; border-left: 4px solid #2d5f5d;">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📌 Conclusion</div>', unsafe_allow_html=True)

st.markdown("""
본 연구는 **머신러닝 기반 유전자 발현 예측 모델**이 새롭게 진단된 다발성 골수종 환자를 
임상적으로 의미 있는 위험군으로 효과적으로 분류할 수 있음을 입증하였습니다.

- ✅ **독립 검증**: TT3 코호트에서 우수한 성능 확인
- ✅ **강한 상관관계**: 예측 위험도와 실제 사망률 간 Spearman Rho = 0.888
- ✅ **생물학적 타당성**: 알려진 MM 바이오마커(SPARC, CD58 등) 포함
- ✅ **임상 적용 가능성**: 0% → 100% 사망률 범위로 명확한 위험 계층화

이러한 결과는 **정밀 종양학(Precision Oncology)** 분야에서 ML 기반 위험 예측 도구가 
실제 임상 의사결정을 지원할 수 있는 잠재력을 보여줍니다.
""")

st.markdown('</div>', unsafe_allow_html=True)
