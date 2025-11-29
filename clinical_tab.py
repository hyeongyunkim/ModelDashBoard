import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def render_clinical_tab() -> None:
    """Clinical Interpretation 탭 렌더링"""

    # ---------------------
    # 📌 Understanding Your Results
    # ---------------------
    st.markdown(
        '<div class="section-title">📋 Understanding Your Results</div>',
        unsafe_allow_html=True,
    )

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

    # ---------------------
    # 🏥 Risk Group Classification (3개군)
    # ---------------------
    st.markdown("### 🏥 Risk Group Classification")
    st.markdown("환자는 Risk Score에 따라 **3개의 위험군**으로 분류됩니다:")

    risk_groups = pd.DataFrame(
        {
            "Risk Group": ["Low Risk", "Medium Risk", "High Risk"],
            "Risk Score Range": ["0.66 - 1.0", "0.33 - 0.66", "0.0 - 0.33"],
            "Expected Survival": ["66-100%", "33-66%", "0-33%"],
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

    # ---------------------
    # 📊 Model Performance
    # ---------------------
    st.markdown("### 📊 Model Performance Metrics")

    col1, col2 = st.columns([1, 1])

    with col1:
        metrics_data = pd.DataFrame(
            {
                "Metric": ["AUC", "MCC", "Recall", "Precision", "F1-Score", "Accuracy"],
                "Value": [0.92, 0.85, 0.89, 0.91, 0.90, 0.88],
            }
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(metrics_data["Metric"], metrics_data["Value"], color="#3d7f7d")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score", fontsize=11, fontweight="bold")
        ax.set_title("Prediction Model Performance", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        for i, v in enumerate(metrics_data["Value"]):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center")

        st.pyplot(fig)

    with col2:
        st.markdown(
            """
#### 성능 지표 설명
- **AUC (0.92)**: 전체 예측 성능 우수  
- **MCC (0.85)**: 불균형 데이터에서도 강건  
- **Recall (0.89)**: 실제 고위험 환자 포착력  
- **Precision (0.91)**: 예측된 고위험 중 실제 고위험 비율  
- **F1-Score (0.90)**: 균형 잡힌 성능  
            """
        )

    # ---------------------
    # 📊 Decile Analysis
    # ---------------------
    st.markdown("### 📊 Decile Analysis Summary")
    st.markdown("독립 검증 데이터셋(TT3, n=214)에서 모델 성능 검증됨.")

    col1, col2 = st.columns([2, 1])

    with col1:
        decile = pd.DataFrame(
            {
                "Decile": list(range(1, 11)),
                "Mortality_Rate": [0, 10, 20, 30, 45, 60, 72, 85, 93, 100],
            }
        )
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(decile["Decile"], decile["Mortality_Rate"], marker="o", linewidth=3, color="#dc3545")
        ax2.fill_between(decile["Decile"], decile["Mortality_Rate"], alpha=0.2, color="#dc3545")
        ax2.set_title("Mortality Rate by Risk Decile", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Risk Decile")
        ax2.set_ylabel("Mortality Rate (%)")
        ax2.set_ylim(-5, 105)
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig2)

    with col2:
        st.markdown(
            """
#### 주요 발견  
- Spearman ρ = **0.888**, p < 0.001  
- 1분위: 0% 사망률  
- 10분위: 100% 사망률  
➡️ 모델의 **임상적 타당성** 입증
            """
        )

    # ---------------------
    # ⚠️ Why High-Risk Patients Matter
    # ---------------------
    st.markdown(
        """
<div class="card">
  <h3>⚠️ Why High-Risk Patients Matter</h3>
  <p><b>고위험 환자 조기 식별</b>은 치료 전략 최적화에 필수적입니다.</p>
  <p><b>1. 치료 강도 결정</b><br>고위험 → 더 강한 치료, 저위험 → 표준 치료</p>
  <p><b>2. 임상시험 참여</b><br>고위험 환자 대상 신약 시험 참여 가능</p>
  <p><b>3. 모니터링 주기</b><br>고위험: 집중 관찰 / 저위험: 정기 체크</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # ---------------------
    # 💡 Clinical Applications
    # ---------------------
    st.markdown(
        """
<div class="card" style="background:#e8f4f3; border-left:4px solid #2d5f5d;">
  <h3>💡 Clinical Applications</h3>
  <p>✅ 진단 시점 위험 평가</p>
  <p>✅ 환자 맞춤형 치료 전략</p>
  <p>✅ 임상 의사결정 지원</p>
  <p>✅ 정밀 종양학 기반 환자 계층화</p>
  <hr>
  <p><b>⚠️ 중요</b>: 본 도구는 임상 결정을 <b>보조</b>하기 위한 것입니다.<br>최종 치료 결정은 전문의 판단이 필요합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )
