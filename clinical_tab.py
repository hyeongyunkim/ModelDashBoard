import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def render_clinical_tab() -> None:
    """Clinical Interpretation 탭"""

    # ========= Section Title =========
    st.markdown(
        '<div class="section-title">📋 Understanding Your Results</div>',
        unsafe_allow_html=True,
    )

    # ========= What is Risk Score =========
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 What is Risk Score?")
    st.markdown(
        """
**Risk Score**는 환자의 **2년 내 사망 확률(0~1)** 을 나타냅니다.

- **0에 가까울수록 → 높은 위험 (낮은 생존율)**  
- **1에 가까울수록 → 낮은 위험 (높은 생존율)**  

이 점수는 **200개 핵심 유전자 발현 패턴**을 기반으로 예측 모델이 계산합니다.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ========= Risk Group Classification (3 groups) =========
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏥 Risk Group Classification")
    st.markdown("환자는 Risk Score를 기반으로 **3개의 위험군**으로 분류됩니다:")

    risk_groups = pd.DataFrame(
        {
            "Risk Group": ["Low Risk", "Medium Risk", "High Risk"],
            "Risk Score Range": ["0.6 - 1.0", "0.3 - 0.6", "0.0 - 0.3"],
            "Expected Survival": ["60–100%", "30–60%", "0–30%"],
            "Clinical Action": [
                "Standard treatment / Regular monitoring",
                "Close observation",
                "Aggressive / Intensive therapy",
            ],
        }
    )

    st.dataframe(risk_groups, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ========= Model Performance =========
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Model Performance Metrics")

    c1, c2 = st.columns([1, 1])

    with c1:
        metrics_data = pd.DataFrame(
            {
                "Metric": ["AUC", "MCC", "Recall", "Precision", "F1-Score", "Accuracy"],
                "Value": [0.92, 0.85, 0.89, 0.91, 0.90, 0.88],
            }
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(metrics_data["Metric"], metrics_data["Value"], color="#3d7f7d")
        ax.set_xlabel("Score", fontsize=11, fontweight="bold")
        ax.set_title("Prediction Model Performance", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis="x")

        # 숫자 라벨
        for i, v in enumerate(metrics_data["Value"]):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=10, fontweight="bold")

        st.pyplot(fig)

    with c2:
        st.markdown(
            """
#### 성능 지표 설명  
**AUC (0.92)** – 모델의 전반적 예측 성능 우수  
**MCC (0.85)** – 불균형 데이터에서도 강건  
**Recall (0.89)** – 실제 고위험 환자 89% 정확히 탐지  
**Precision (0.91)** – 고위험으로 예측된 환자 중 91%가 실제 고위험  
**F1-score (0.90)** – Precision·Recall 균형 우수
"""
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ========= Decile Analysis =========
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Decile Analysis Summary")
    st.markdown("본 모델은 **독립 검증 데이터셋(TT3, n=214)**에서 임상적 타당성을 입증했습니다.")

    c1, c2 = st.columns([2, 1])

    with c1:
        decile_df = pd.DataFrame(
            {
                "Decile": list(range(1, 11)),
                "Mortality_Rate": [0, 8, 18, 28, 40, 55, 70, 85, 95, 100],
            }
        )

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(
            decile_df["Decile"],
            decile_df["Mortality_Rate"],
            marker="o",
            linewidth=3,
            markersize=10,
            color="#dc3545",
        )
        ax2.fill_between(decile_df["Decile"], decile_df["Mortality_Rate"], alpha=0.2, color="#dc3545")
        ax2.set_xlabel("Risk Decile (1 = Lowest Risk, 10 = Highest)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Mortality Rate (%)", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    with c2:
        st.markdown(
            """
**Spearman Rho = 0.888 (p < 0.001)**  

- 최저 위험군 1분위: **0% 사망률**  
- 최고 위험군 10분위: **100% 사망률**  

➡️ 예측 위험도와 실제 사망률 간 **강한 단조적 상관관계**  
"""
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ========= Contributing Genes =========
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧬 Top 10 Contributing Genes")

    c1, c2 = st.columns([2, 1])

    with c1:
        df_genes = pd.DataFrame(
            {
                "Gene": ["SPARC", "C2orf74", "FAM105A", "AKR1C3", "EPS8L3", "IL2", "SNX2", "LOC100506125", "CD58", "ARHGEF37"],
                "Importance": [0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05],
                "Known": ["Yes", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No"],
            }
        )

        colors = ["#dc3545" if k == "Yes" else "#3d7f7d" for k in df_genes["Known"]]

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.barh(df_genes["Gene"], df_genes["Importance"], color=colors)
        ax3.set_xlabel("Feature Importance", fontsize=12, fontweight="bold")
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis="x")

        for i, imp in enumerate(df_genes["Importance"]):
            ax3.text(imp + 0.003, i, f"{imp:.3f}", va="center", fontsize=10)

        legend_items = [
            Patch(facecolor="#dc3545", label="Known Biomarker"),
            Patch(facecolor="#3d7f7d", label="Other Gene"),
        ]
        ax3.legend(handles=legend_items, loc="lower right")

        st.pyplot(fig3)

    with c2:
        st.markdown(
            """
#### Known Biomarkers  
**SPARC** ⭐ – MM 바이오마커  
**CD58** ⭐ – 면역 관련  
**IL2** ⭐ – T세포 활성화  

➡️ 모델의 **생물학적 타당성** 확인  
"""
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ========= Why High-Risk Matters =========
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚠️ Why High-Risk Patients Matter")
    st.markdown(
        """
고위험 환자 조기 식별은 치료 전략에서 매우 중요합니다:

- **고위험 → 적극적 초기 치료 필요**  
- **중위험 → 면밀한 관찰 필요**  
- **저위험 → 표준 치료/모니터링으로 충분**  

고위험 환자는 신약 임상시험 참여 가능성이 높으며,  
예후 상담 및 치료 계획 수립에도 핵심 기준이 됩니다.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ========= Clinical Applications =========
    st.markdown('<div class="card" style="background:#e8f4f3; border-left:4px solid #2d5f5d;">', unsafe_allow_html=True)
    st.markdown(
        """
### 💡 Clinical Applications

✔️ **진단 시점 위험 평가**  
✔️ **개인 맞춤형 치료 전략**  
✔️ **임상 의사결정 지원**  
✔️ **정밀 종양학 적용**  

---
⚠️ 최종 치료 결정은 반드시 전문의 판단 하에 이루어져야 합니다.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)
