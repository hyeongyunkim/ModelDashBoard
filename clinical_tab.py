import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ================================
# 🔧 CSS — 박스가 따로 노는 현상 완전 해결
# ================================
st.markdown(
    """
<style>
    .card {
        background: white;
        padding: 1.8rem 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        margin-top: 0rem !important;       /* 박스 위 여백 제거 */
        margin-bottom: 1.5rem !important;  /* 아래 여백 통일 */
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2d5f5d;
        margin-bottom: 0.8rem !important;  /* 제목 아래 간격 최적화 */
        border-left: 4px solid #3d7f7d;
        padding-left: 1rem;
    }
</style>
""",
    unsafe_allow_html=True
)


# ================================================================
#                Clinical Interpretation 탭 렌더링 함수
# ================================================================
def render_clinical_tab() -> None:

    # -------------------- Title -------------------
    st.markdown('<div class="section-title">📋 Understanding Your Results</div>',
                unsafe_allow_html=True)

    # -------------------- Risk Score 설명 --------------------
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 What is Risk Score?")
        st.markdown(
            """
**Risk Score**는 환자의 **2년 내 사망 확률**을 나타냅니다.

- **0에 가까울수록** → 낮은 사망 위험 (높은 생존율)  
- **1에 가까울수록** → 높은 사망 위험 (낮은 생존율)

이 점수는 200개의 핵심 유전자 발현 패턴을 기반으로 예측 모델이 계산합니다.
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Risk Group 설명 (3단계 버전) --------------------
    st.markdown('<div class="section-title">🏥 Risk Group Classification</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("환자는 Risk Score에 따라 **3가지 위험군**으로 분류됩니다:")

    risk_groups = pd.DataFrame(
        {
            "Risk Group": ["Low Risk", "Medium Risk", "High Risk"],
            "Risk Score Range": ["0.0 - 0.33", "0.33 - 0.66", "0.66 - 1.0"],
            "Expected Survival": ["67–100%", "34–66%", "0–33%"],
            "Clinical Action": [
                "Standard treatment",
                "Close monitoring",
                "Aggressive / intensive therapy",
            ],
        }
    )

    st.dataframe(risk_groups, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Model Performance --------------------
    st.markdown('<div class="section-title">📊 Model Performance Metrics</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])

    with c1:
        metrics_data = pd.DataFrame(
            {
                "Metric": ["AUC", "MCC", "Recall", "Precision", "F1-Score", "Accuracy"],
                "Value": [0.92, 0.85, 0.89, 0.91, 0.90, 0.88],
            }
        )

        fig5, ax5 = plt.subplots(figsize=(7, 4))
        ax5.barh(metrics_data["Metric"], metrics_data["Value"], color="#3d7f7d")
        ax5.set_xlabel("Score", fontsize=11, fontweight="bold")
        ax5.set_xlim(0, 1)
        ax5.grid(True, axis="x", alpha=0.3)

        for i, v in enumerate(metrics_data["Value"]):
            ax5.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=10)

        st.pyplot(fig5)

    with c2:
        st.markdown(
            """
**AUC (0.92)** – 전체 예측 성능 우수  
**MCC (0.85)** – 불균형 데이터에서도 안정적  
**Recall (0.89)** – 실제 고위험 환자 잘 잡음  
**Precision (0.91)** – 예측한 고위험 환자 대부분이 실제 고위험  
**F1-Score (0.90)** – Precision + Recall 균형 우수
"""
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Decile Analysis --------------------
    st.markdown('<div class="section-title">📈 Decile Analysis Summary</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("모델은 독립 검증 세트(TT3, n=214)에서 뛰어난 성능을 보였습니다.")

    c1, c2 = st.columns([1.5, 1])

    with c1:
        decile_data = pd.DataFrame(
            {"Decile": list(range(1, 11)),
             "Mortality_Rate": [0, 10, 18, 28, 42, 58, 72, 85, 94, 100]}
        )

        fig6, ax6 = plt.subplots(figsize=(8, 5))
        ax6.plot(decile_data["Decile"], decile_data["Mortality_Rate"],
                 marker="o", linewidth=3, color="#dc3545")
        ax6.fill_between(decile_data["Decile"], decile_data["Mortality_Rate"],
                         alpha=0.2, color="#dc3545")

        ax6.set_xlabel("Risk Decile (1 = Lowest, 10 = Highest)")
        ax6.set_ylabel("Mortality Rate (%)")
        ax6.set_ylim(-5, 105)
        ax6.grid(True, alpha=0.3)

        st.pyplot(fig6)

    with c2:
        st.markdown(
            """
- **Spearman’s Rho = 0.888 (p < 0.001)**  
- 1분위: **0% 사망률**  
- 10분위: **100% 사망률**  

➡️ 위험 점수와 실제 사망률 간 매우 강한 상관성을 보여 모델의 임상적 타당성을 입증합니다.
"""
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- Gene Importance --------------------
    st.markdown('<div class="section-title">🧬 Top 10 Contributing Genes</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 1])

    with c1:
        gene_importance = pd.DataFrame(
            {
                "Gene": ["SPARC", "C2orf74", "FAM105A", "AKR1C3", "EPS8L3",
                         "IL2", "SNX2", "LOC100506125", "CD58", "ARHGEF37"],
                "Importance": [0.12, 0.10, 0.09, 0.08, 0.08,
                               0.07, 0.07, 0.06, 0.06, 0.05],
                "Known_Biomarker": ["Yes", "No", "No", "No", "No",
                                    "Yes", "No", "No", "Yes", "No"],
            }
        )

        fig7, ax7 = plt.subplots(figsize=(8, 5))
        colors = ["#dc3545" if x == "Yes" else "#3d7f7d"
                  for x in gene_importance["Known_Biomarker"]]

        ax7.barh(gene_importance["Gene"], gene_importance["Importance"], color=colors)
        ax7.invert_yaxis()
        ax7.set_xlabel("Feature Importance")

        for i, v in enumerate(gene_importance["Importance"]):
            ax7.text(v + 0.002, i, f"{v:.3f}")

        legend_elems = [
            Patch(facecolor="#dc3545", label="Known MM Biomarker"),
            Patch(facecolor="#3d7f7d", label="Other Gene"),
        ]
        ax7.legend(handles=legend_elems, loc="lower right")

        st.pyplot(fig7)

    with c2:
        st.markdown(
            """
### Known Biomarkers
**SPARC** – Extracellular matrix protein  
**IL2** – Immune activation  
**CD58** – T-cell adhesion molecule  

➡️ 모델이 생물학적으로 타당한 유전자를 반영하고 있음
"""
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- 고위험군 중요성 --------------------
    st.markdown('<div class="section-title">⚠️ Why High-Risk Patients Matter</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        """
- 고위험 환자는 **더 적극적인 치료 전략** 필요  
- 임상시험 참여 가능성이 높음  
- 예후 상담 및 모니터링 계획에 중요한 기준 제공  
"""
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- 임상 활용 --------------------
    st.markdown('<div class="section-title">💡 Clinical Applications</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="card" style="background:#e8f4f3;border-left:4px solid #2d5f5d;">',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
- 진단 시점에 즉시 위험 예측  
- 위험군 기반 맞춤형 치료 전략  
- 객관적 예측 모델을 통한 의사결정 지원  
- 정밀 종양학 실현  
"""
    )

    st.markdown("</div>", unsafe_allow_html=True)
