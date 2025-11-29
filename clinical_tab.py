import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def render_clinical_tab() -> None:
    """Clinical Interpretation 탭을 렌더링하는 함수."""

    # 섹션 타이틀 공통
    st.markdown(
        '<div class="section-title">📋 Understanding Your Results</div>',
        unsafe_allow_html=True,
    )

    # ---------- Risk Score 설명 (텍스트 + 카드 안에 다 넣음) ----------
    st.markdown(
        """
<div class="card">
  <h3>🎯 What is Risk Score?</h3>
  <p><b>Risk Score</b>는 환자의 <b>2년 내 사망 확률</b>을 나타냅니다.</p>
  <ul>
    <li><b>0에 가까울수록</b>: 낮은 사망 위험 (높은 생존율)</li>
    <li><b>1에 가까울수록</b>: 높은 사망 위험 (낮은 생존율)</li>
  </ul>
  <p>이 점수는 200개의 핵심 유전자 발현 패턴을 예측 모델이 분석하여 계산됩니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # ---------- Risk Group 설명 (3개 위험군) ----------
    st.markdown("### 🏥 Risk Group Classification")
    st.markdown("환자는 Risk Score를 기반으로 **3개의 위험군**으로 분류됩니다:")

    risk_groups = pd.DataFrame(
        {
            "Risk Group": [
                "Low Risk",
                "Medium Risk",
                "High Risk",
            ],
            "Risk Score Range": [
                "0.66 - 1.0",
                "0.33 - 0.66",
                "0.0 - 0.33",
            ],
            "Expected Survival": [
                "66-100%",
                "33-66%",
                "0-33%",
            ],
            "Clinical Action": [
                "Standard treatment / 정기 추적",
                "Close monitoring / 치료 전략 조정",
                "Aggressive / intensive therapy",
            ],
        }
    )

    st.dataframe(
        risk_groups,
        use_container_width=True,
        hide_index=True,
    )

    # ---------- 모델 성능 (그래프 섹션: 카드 래퍼 제거) ----------
    st.markdown("### 📊 Model Performance Metrics")

    c1, c2 = st.columns([1, 1])

    with c1:
        metrics_data = pd.DataFrame(
            {
                "Metric": [
                    "AUC",
                    "MCC",
                    "Recall",
                    "Precision",
                    "F1-Score",
                    "Accuracy",
                ],
                "Value": [0.92, 0.85, 0.89, 0.91, 0.90, 0.88],
            }
        )

        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.barh(metrics_data["Metric"], metrics_data["Value"], color="#3d7f7d")
        ax5.set_xlabel("Score", fontsize=11, fontweight="bold")
        ax5.set_title(
            "Prediction Model Performance",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax5.set_xlim(0, 1)
        ax5.grid(True, alpha=0.3, axis="x")

        for i, value in enumerate(metrics_data["Value"]):
            ax5.text(
                value + 0.02,
                i,
                f"{value:.2f}",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        st.pyplot(fig5)

    with c2:
        st.markdown(
            """
#### 성능 지표 설명

**AUC (0.92)**: 모델의 전반적인 분류 성능이 매우 우수함  

**MCC (0.85)**: 불균형 데이터에서도 강건한 예측력  

**Recall (0.89)**: 실제 고위험 환자의 89%를 정확히 포착  

**Precision (0.91)**: 고위험으로 예측한 환자 중 91%가 실제 고위험  

**F1-Score (0.90)**: Precision과 Recall의 균형잡힌 성능
"""
        )

    # ---------- Decile 분석 (그래프 섹션) ----------
    st.markdown("### 📊 Decile Analysis Summary")
    st.markdown(
        "본 모델은 **독립 검증 데이터셋(TT3, n=214)**에서 뛰어난 성능을 입증했습니다."
    )

    c1, c2 = st.columns([2, 1])

    with c1:
        decile_data = pd.DataFrame(
            {
                "Decile": list(range(1, 11)),
                "Mortality_Rate": [0, 10, 20, 30, 45, 60, 72, 85, 93, 100],
            }
        )

        fig6, ax6 = plt.subplots(figsize=(10, 6))
        ax6.plot(
            decile_data["Decile"],
            decile_data["Mortality_Rate"],
            marker="o",
            linewidth=3,
            markersize=12,
            color="#dc3545",
        )
        ax6.fill_between(
            decile_data["Decile"],
            decile_data["Mortality_Rate"],
            alpha=0.2,
            color="#dc3545",
        )
        ax6.set_xlabel(
            "Risk Decile (1=Lowest, 10=Highest)",
            fontsize=12,
            fontweight="bold",
        )
        ax6.set_ylabel(
            "Mortality Rate (%)",
            fontsize=12,
            fontweight="bold",
        )
        ax6.set_title(
            "Mortality Rate by Risk Decile",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax6.grid(True, alpha=0.3)
        ax6.set_xticks(range(1, 11))
        ax6.set_ylim(-5, 105)

        plt.tight_layout()
        st.pyplot(fig6)

    with c2:
        st.markdown(
            """
#### 주요 발견

**Spearman's Rho = 0.888**  (p < 0.001)

- 1분위: **0%** 사망률  
- 10분위: **100%** 사망률  

➡️ 예측 위험도와 실제 사망률 간 **강한 단조적 상관관계** 확인  

이는 모델의 **임상적 타당성**을 입증합니다.
"""
        )

    # ---------- Top 10 유전자 (그래프 섹션) ----------
    st.markdown("### 🧬 Top 10 Contributing Genes")

    c1, c2 = st.columns([2, 1])

    with c1:
        gene_importance = pd.DataFrame(
            {
                "Gene": [
                    "SPARC",
                    "C2orf74",
                    "FAM105A",
                    "AKR1C3",
                    "EPS8L3",
                    "IL2",
                    "SNX2",
                    "LOC100506125",
                    "CD58",
                    "ARHGEF37",
                ],
                "Importance": [
                    0.12,
                    0.10,
                    0.09,
                    0.08,
                    0.08,
                    0.07,
                    0.07,
                    0.06,
                    0.06,
                    0.05,
                ],
                "Known_Biomarker": [
                    "Yes",
                    "No",
                    "No",
                    "No",
                    "No",
                    "Yes",
                    "No",
                    "No",
                    "Yes",
                    "No",
                ],
            }
        )

        fig7, ax7 = plt.subplots(figsize=(10, 6))
        colors_genes = [
            "#dc3545" if x == "Yes" else "#3d7f7d"
            for x in gene_importance["Known_Biomarker"]
        ]
        ax7.barh(
            gene_importance["Gene"],
            gene_importance["Importance"],
            color=colors_genes,
        )
        ax7.set_xlabel("Feature Importance", fontsize=12, fontweight="bold")
        ax7.set_title(
            "Top 10 Contributing Genes",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax7.invert_yaxis()
        ax7.grid(True, alpha=0.3, axis="x")

        for i, imp in enumerate(gene_importance["Importance"]):
            ax7.text(
                imp + 0.003,
                i,
                f"{imp:.3f}",
                va="center",
                fontsize=10,
            )

        legend_elements = [
            Patch(facecolor="#dc3545", label="Known MM Biomarker"),
            Patch(facecolor="#3d7f7d", label="Other Gene"),
        ]
        ax7.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        st.pyplot(fig7)

    with c2:
        st.markdown(
            """
#### Known Biomarkers

**SPARC** ⭐  
- MM 바이오마커  
- 세포외 기질 단백질  

**CD58** ⭐  
- 면역 조절 관련  
- MM 예후 마커  

**IL2** ⭐  
- 면역 반응 관련  
- T세포 활성화  

➡️ 모델의 **생물학적 타당성** 확보
"""
        )

    # ---------- 고위험군의 중요성 (텍스트 카드) ----------
    st.markdown(
        """
<div class="card">
  <h3>⚠️ Why High-Risk Patients Matter</h3>
  <p><b>고위험 환자 조기 식별</b>은 다발성 골수종 치료에서 매우 중요합니다:</p>
  <p><b>1. 치료 강도 결정</b><br>
     - 고위험 → 더 적극적인 초기 치료<br>
     - 저위험 → 부작용 최소화한 표준 치료</p>
  <p><b>2. 임상시험 참여</b><br>
     - 고위험군 대상 신약 임상시험<br>
     - 맞춤형 치료법 개발</p>
  <p><b>3. 모니터링 주기</b><br>
     - 고위험: 집중 추적 관찰<br>
     - 저위험: 정기 검진</p>
  <p><b>4. 예후 상담</b><br>
     - 정확한 예후 정보 제공<br>
     - 치료 계획 수립 지원</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # ---------- 임상 활용 (배경색 카드 한 번에) ----------
    st.markdown(
        """
<div class="card" style="background:#e8f4f3; border-left:4px solid #2d5f5d;">
  <h3>💡 Clinical Applications</h3>
  <p>✅ <b>진단 시점 위험 평가</b> - 새로 진단된 MM 환자의 예후 예측</p>
  <p>✅ <b>개인 맞춤형 치료</b> - 위험군별 차별화된 치료 프로토콜</p>
  <p>✅ <b>임상 의사결정 지원</b> - 200개 유전자 기반 객관적 예측</p>
  <p>✅ <b>정밀 종양학 실현</b> - 분자 수준의 환자 계층화</p>
  <hr>
  <p><b>⚠️ 중요</b>: 이 도구는 임상 의사결정을 <b>보조</b>하는 목적으로 개발되었으며,<br>
     최종 치료 결정은 반드시 전문의의 종합적인 판단 하에 이루어져야 합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )
