import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def render_clinical_tab() -> None:
    """Clinical Interpretation 탭"""

    # =========================
    # 1. Understanding Your Results
    # =========================
    st.markdown(
        '<div class="section-title">📋 Understanding Your Results</div>',
        unsafe_allow_html=True,
    )

    # 카드 + 내용 한 번에 (따로 안 나눔)
    st.markdown(
        """
<div class="card">
  <h3>🎯 What is Risk Score?</h3>
  <p><b>Risk Score</b>는 환자의 <b>2년 내 사망 확률(0~1)</b>을 의미합니다.</p>
  <ul>
    <li><b>0에 가까울수록</b> → 높은 사망 위험 (낮은 생존율)</li>
    <li><b>1에 가까울수록</b> → 낮은 사망 위험 (높은 생존율)</li>
  </ul>
  <p>이 점수는 <b>200개 핵심 유전자 발현 패턴</b>을 기반으로 예측 모델이 계산합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # =========================
    # 2. Risk Group Classification (3단계)
    # =========================
    st.markdown(
        '<div class="section-title">🏥 Risk Group Classification</div>',
        unsafe_allow_html=True,
    )

    # 여기서는 표까지 전부 HTML로 카드 안에 넣어버림
    st.markdown(
        """
<div class="card">
  <p>환자는 Risk Score에 따라 <b>3가지 위험군</b>으로 분류됩니다.</p>
  <table style="width:100%; border-collapse:collapse; font-size:0.9rem;">
    <thead>
      <tr style="background-color:#f8f9fa;">
        <th style="padding:8px; border:1px solid #e9ecef; text-align:left;">Risk Group</th>
        <th style="padding:8px; border:1px solid #e9ecef; text-align:left;">Risk Score Range</th>
        <th style="padding:8px; border:1px solid #e9ecef; text-align:left;">Expected Survival</th>
        <th style="padding:8px; border:1px solid #e9ecef; text-align:left;">Clinical Action</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:8px; border:1px solid #e9ecef;">Low Risk</td>
        <td style="padding:8px; border:1px solid #e9ecef;">0.66 – 1.00</td>
        <td style="padding:8px; border:1px solid #e9ecef;">66–100%</td>
        <td style="padding:8px; border:1px solid #e9ecef;">Standard treatment / 정기 추적</td>
      </tr>
      <tr style="background-color:#f8f9fa;">
        <td style="padding:8px; border:1px solid #e9ecef;">Medium Risk</td>
        <td style="padding:8px; border:1px solid #e9ecef;">0.33 – 0.66</td>
        <td style="padding:8px; border:1px solid #e9ecef;">33–66%</td>
        <td style="padding:8px; border:1px solid #e9ecef;">Closer monitoring / 치료 전략 조정</td>
      </tr>
      <tr>
        <td style="padding:8px; border:1px solid #e9ecef;">High Risk</td>
        <td style="padding:8px; border:1px solid #e9ecef;">0.00 – 0.33</td>
        <td style="padding:8px; border:1px solid #e9ecef;">0–33%</td>
        <td style="padding:8px; border:1px solid #e9ecef;">Aggressive / intensive therapy</td>
      </tr>
    </tbody>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )

    # =========================
    # 3. Model Performance Metrics (차트 — 카드 안 쓰지 않음)
    # =========================
    st.markdown(
        '<div class="section-title">📊 Model Performance Metrics</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.3, 1])

    with col1:
        metrics_data = pd.DataFrame(
            {
                "Metric": ["AUC", "MCC", "Recall", "Precision", "F1-Score", "Accuracy"],
                "Value": [0.92, 0.85, 0.89, 0.91, 0.90, 0.88],
            }
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(metrics_data["Metric"], metrics_data["Value"], color="#3d7f7d")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.grid(True, axis="x", alpha=0.3)
        for i, v in enumerate(metrics_data["Value"]):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=10)
        st.pyplot(fig)

    with col2:
        st.markdown(
            """
**AUC (0.92)** – 전체 예측 성능 우수  
**MCC (0.85)** – 불균형 데이터에서도 안정적  
**Recall (0.89)** – 실제 고위험 환자 잘 포착  
**Precision (0.91)** – 고위험으로 예측된 환자 중 91%가 실제 고위험  
**F1-Score (0.90)** – Precision·Recall 균형 우수  
"""
        )

    # =========================
    # 4. Decile Analysis
    # =========================
    st.markdown(
        '<div class="section-title">📈 Decile Analysis Summary</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.5, 1])

    with c1:
        decile_df = pd.DataFrame(
            {
                "Decile": list(range(1, 11)),
                "Mortality_Rate": [0, 8, 18, 28, 40, 55, 70, 85, 95, 100],
            }
        )
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(
            decile_df["Decile"],
            decile_df["Mortality_Rate"],
            marker="o",
            linewidth=3,
            color="#dc3545",
        )
        ax2.fill_between(
            decile_df["Decile"],
            decile_df["Mortality_Rate"],
            alpha=0.2,
            color="#dc3545",
        )
        ax2.set_xlabel("Risk Decile (1 = Lowest, 10 = Highest)")
        ax2.set_ylabel("Mortality Rate (%)")
        ax2.set_ylim(-5, 105)
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    with c2:
        st.markdown(
            """
**Spearman Rho = 0.888 (p < 0.001)**  

- 최저 위험군(1분위): **0% 사망률**  
- 최고 위험군(10분위): **100% 사망률**  

➡️ 예측 위험도와 실제 사망률 간 **강한 단조적 상관관계** 확인  
→ 모델의 **임상적 타당성**을 뒷받침  
"""
        )

    # =========================
    # 5. Top 10 Contributing Genes
    # =========================
    st.markdown(
        '<div class="section-title">🧬 Top 10 Contributing Genes</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.5, 1])

    with c1:
        gene_df = pd.DataFrame(
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
                "Importance": [0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05],
                "Known": ["Yes", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No"],
            }
        )
        colors = ["#dc3545" if k == "Yes" else "#3d7f7d" for k in gene_df["Known"]]

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.barh(gene_df["Gene"], gene_df["Importance"], color=colors)
        ax3.invert_yaxis()
        ax3.set_xlabel("Feature Importance")
        ax3.grid(True, axis="x", alpha=0.3)
        for i, v in enumerate(gene_df["Importance"]):
            ax3.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
        legend_items = [
            Patch(facecolor="#dc3545", label="Known MM Biomarker"),
            Patch(facecolor="#3d7f7d", label="Other Gene"),
        ]
        ax3.legend(handles=legend_items, loc="lower right")
        st.pyplot(fig3)

    with c2:
        st.markdown(
            """
**SPARC** – MM 관련 바이오마커  
**IL2** – T세포 활성 / 면역 반응  
**CD58** – 면역 세포 결합 관련  

➡️ 모델이 실제 알려진 바이오마커를 잘 반영하고 있어  
   **생물학적 타당성**을 뒷받침합니다.  
"""
        )

    # =========================
    # 6. Why High-Risk Patients Matter (텍스트 카드)
    # =========================
    st.markdown(
        """
<div class="section-title">⚠️ Why High-Risk Patients Matter</div>
<div class="card">
  <p><b>고위험 환자 조기 식별</b>은 치료 전략에서 매우 중요합니다.</p>
  <ul>
    <li>초기부터 더 공격적인 치료 여부 결정</li>
    <li>신약 임상시험 참여 대상 선정</li>
    <li>추적 관찰 주기(visit interval) 설정</li>
    <li>예후 상담 및 환자·보호자 교육</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    # =========================
    # 7. Clinical Applications (텍스트 카드)
    # =========================
    st.markdown(
        """
<div class="section-title">💡 Clinical Applications</div>
<div class="card" style="background:#e8f4f3; border-left:4px solid #2d5f5d;">
  <ul>
    <li><b>진단 시점 위험 평가</b> – 새로 진단된 MM 환자의 예후 예측</li>
    <li><b>개인 맞춤형 치료</b> – 위험군별 차별화된 치료 전략 설계</li>
    <li><b>임상 의사결정 지원</b> – 정량적 Risk Score 기반 근거 제공</li>
    <li><b>정밀 종양학 구현</b> – 분자 프로파일 기반 환자 계층화</li>
  </ul>
  <hr>
  <p><b>⚠️ 주의</b>: 이 도구는 <b>의사의 판단을 보조</b>하기 위한 것이며,  
     최종 치료 결정은 반드시 담당 전문의의 임상적 판단에 따라야 합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )
