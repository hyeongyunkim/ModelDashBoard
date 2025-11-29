import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from clinical_tab import render_clinical_tab  # 두 번째 탭 렌더링 함수


# -------------------------------------------------------
# 페이지 설정
# -------------------------------------------------------
st.set_page_config(
    page_title="MM Risk Predictor",
    page_icon="🧬",
    layout="wide",
)

# -------------------------------------------------------
# 커스텀 CSS
# -------------------------------------------------------
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

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
st.markdown(
    """
<div class="header-container">
    <div class="header-title">🧬 MM Risk Predictor</div>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
        Gene expression-based Multiple Myeloma Prognosis Prediction
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# 탭 생성
# -------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Predict My Sample", "📋 Clinical Interpretation"])

# =======================================================
# 탭 1: Predict My Sample
# =======================================================
with tab1:
    st.markdown(
        '<div class="section-title">📁 Upload Patient Data</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload CSV file with gene expression data",
        type=["csv"],
    )

    if uploaded is None:
        st.markdown(
            """
        <div class="upload-container">
            <div style="font-size: 4rem; color: #3d7f7d; margin-bottom: 1rem;">📁</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #2d5f5d; margin-bottom: 0.5rem;">
                Upload Gene Expression Data
            </div>
            <div style="font-size: 1rem; color: #6c757d;">
                CSV file with 200 gene features required
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.info(
            "📋 **Required format**: CSV file with 200 gene expression "
            "features matching the model's feature set"
        )

    else:
        try:
            user_df = pd.read_csv(uploaded)

            # ---------------- Data Validation ----------------
            st.markdown(
                '<div class="section-title">✅ Data Validation</div>',
                unsafe_allow_html=True,
            )

            missing_features = set(feature_cols) - set(user_df.columns)
            extra_features = set(user_df.columns) - set(feature_cols)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Uploaded Samples", len(user_df))
            with c2:
                st.metric("Required Features", len(feature_cols))
            with c3:
                st.metric(
                    "Matched Features",
                    len(set(feature_cols) & set(user_df.columns)),
                )

            if missing_features:
                st.error(f"❌ Missing {len(missing_features)} required features")
                with st.expander("Show missing features"):
                    st.write(list(missing_features)[:10])
                st.stop()

            if extra_features:
                st.warning(
                    f"⚠️ Found {len(extra_features)} extra columns (will be ignored)"
                )

            st.success("✅ All required features found! Ready for prediction.")

            # ---------------- 예측 함수 ----------------
            def run_prediction(df: pd.DataFrame) -> pd.DataFrame:
                df = df.copy()

                # 1) 이미 Risk_Score / Risk_Group가 있는 샘플 CSV
                if {"Risk_Score", "Risk_Group"}.issubset(df.columns):
                    if "Patient_ID" in df.columns:
                        patient_ids = df["Patient_ID"].astype(str).tolist()
                    else:
                        patient_ids = [
                            f"MM-{str(i + 1).zfill(3)}" for i in range(len(df))
                        ]

                    result_df = pd.DataFrame(
                        {
                            "Patient_ID": patient_ids,
                            "Risk_Score": df["Risk_Score"].astype(float),
                            "Risk_Group": df["Risk_Group"].astype(str),
                        }
                    )

                    if "Survival_Rate" in df.columns:
                        result_df["Survival_Rate"] = df["Survival_Rate"].astype(float)
                    else:
                        result_df["Survival_Rate"] = (
                            1 - result_df["Risk_Score"]
                        ) * 100  # 생존율(%)

                    return result_df

                # 2) 일반 데이터 → 모델로 예측
                df = df[feature_cols]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df)

                risk = model.predict_proba(X_scaled)[:, 1]  # 사망 확률(0~1)

                def get_risk_group(score: float) -> str:
                    if score < 0.2:
                        return "Very High Risk"
                    if score < 0.4:
                        return "High Risk"
                    if score < 0.6:
                        return "Medium Risk"
                    if score < 0.8:
                        return "Low Risk"
                    return "Very Low Risk"

                result_df = pd.DataFrame(
                    {
                        "Patient_ID": [
                            f"MM-{str(i + 1).zfill(3)}" for i in range(len(risk))
                        ],
                        "Risk_Score": risk,
                        "Risk_Group": [get_risk_group(r) for r in risk],
                        "Survival_Rate": [(1 - r) * 100 for r in risk],
                    }
                )
                return result_df

            # ---------------- 예측 실행 ----------------
            st.markdown(
                '<div class="section-title">🔬 Prediction Results</div>',
                unsafe_allow_html=True,
            )

            result_df = run_prediction(user_df)

            # --------- 상단 요약 카드 ---------
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-number">{len(result_df)}</div>
                        <div class="stat-label">Total Patients</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                high_risk = len(
                    result_df[
                        result_df["Risk_Group"].isin(
                            ["High Risk", "Very High Risk"]
                        )
                    ]
                )
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-number" style="color:#dc3545;">{high_risk}</div>
                        <div class="stat-label">High Risk</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                medium_risk = len(
                    result_df[result_df["Risk_Group"] == "Medium Risk"]
                )
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-number" style="color:#ffc107;">{medium_risk}</div>
                        <div class="stat-label">Medium Risk</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c4:
                low_risk = len(
                    result_df[
                        result_df["Risk_Group"].isin(
                            ["Low Risk", "Very Low Risk"]
                        )
                    ]
                )
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-number" style="color:#28a745;">{low_risk}</div>
                        <div class="stat-label">Low Risk</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # --------- 전체 결과 테이블 ---------
            st.markdown("### 📋 Patient-wise Results")

            def color_risk_group(val: str) -> str:
                colors = {
                    "Very High Risk": "background-color: #f5c6cb",
                    "High Risk": "background-color: #f8d7da",
                    "Medium Risk": "background-color: #fff3cd",
                    "Low Risk": "background-color: #d1ecf1",
                    "Very Low Risk": "background-color: #d4edda",
                }
                return colors.get(val, "")

            display_df = result_df[
                ["Patient_ID", "Survival_Rate", "Risk_Group", "Risk_Score"]
            ].copy()
            display_df["Survival_Rate"] = display_df["Survival_Rate"].apply(
                lambda x: f"{x:.1f}%"
            )
            display_df["Risk_Score"] = display_df["Risk_Score"].apply(
                lambda x: f"{x:.3f}"
            )

            styled_df = display_df.style.applymap(
                color_risk_group,
                subset=["Risk_Group"],
            )
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=300,
            )

            # --------- 시각화 ---------
            st.markdown("### 📊 Visualizations")

            v1, v2 = st.columns(2)

            # 히스토그램
            with v1:
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                ax1.hist(
                    result_df["Risk_Score"],
                    bins=20,
                    color="#3d7f7d",
                    edgecolor="white",
                    alpha=0.7,
                )
                mean_score = result_df["Risk_Score"].mean()
                ax1.axvline(
                    mean_score,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_score:.3f}",
                )
                ax1.set_xlabel(
                    "Risk Score (Death Probability)",
                    fontsize=11,
                    fontweight="bold",
                )
                ax1.set_ylabel(
                    "Number of Patients",
                    fontsize=11,
                    fontweight="bold",
                )
                ax1.set_title(
                    "Risk Score Distribution",
                    fontsize=13,
                    fontweight="bold",
                    pad=15,
                )
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig1)

            # 박스플롯
            with v2:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                risk_order = [
                    "Very Low Risk",
                    "Low Risk",
                    "Medium Risk",
                    "High Risk",
                    "Very High Risk",
                ]
                result_df["Risk_Group_Cat"] = pd.Categorical(
                    result_df["Risk_Group"],
                    categories=risk_order,
                    ordered=True,
                )
                result_sorted = result_df.sort_values("Risk_Group_Cat")

                sns.boxplot(
                    x="Risk_Group_Cat",
                    y="Risk_Score",
                    data=result_sorted,
                    ax=ax2,
                    palette="RdYlGn_r",
                )
                ax2.set_xlabel("Risk Group", fontsize=11, fontweight="bold")
                ax2.set_ylabel("Risk Score", fontsize=11, fontweight="bold")
                ax2.set_title(
                    "Risk Score by Risk Group",
                    fontsize=13,
                    fontweight="bold",
                    pad=15,
                )
                plt.xticks(rotation=45, ha="right")
                ax2.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()
                st.pyplot(fig2)

            # 두 번째 줄 그래프
            v3, v4 = st.columns(2)

            # 위험군 분포 막대그래프
            with v3:
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                risk_counts = result_df["Risk_Group"].value_counts()
                risk_counts = risk_counts.reindex(risk_order, fill_value=0)

                colors_bar = [
                    "#28a745",
                    "#17a2b8",
                    "#ffc107",
                    "#fd7e14",
                    "#dc3545",
                ]

                bars = ax3.bar(
                    range(len(risk_counts)),
                    risk_counts.values,
                    color=colors_bar,
                    edgecolor="white",
                    linewidth=1.5,
                )
                ax3.set_xticks(range(len(risk_counts)))
                ax3.set_xticklabels(
                    risk_counts.index,
                    rotation=45,
                    ha="right",
                    fontsize=9,
                )
                ax3.set_ylabel(
                    "Number of Patients",
                    fontsize=11,
                    fontweight="bold",
                )
                ax3.set_title(
                    "Risk Group Distribution",
                    fontsize=13,
                    fontweight="bold",
                    pad=15,
                )
                ax3.grid(True, alpha=0.3, axis="y")

                max_count = max(risk_counts.values) if len(risk_counts) > 0 else 0
                for bar, count in zip(bars, risk_counts.values):
                    if count > 0:
                        percentage = count / len(result_df) * 100
                        ax3.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max_count * 0.02,
                            f"{count}\n({percentage:.1f}%)",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                        )

                plt.tight_layout()
                st.pyplot(fig3)

            # 개별 환자 점 그래프
            with v4:
                fig4, ax4 = plt.subplots(figsize=(8, 5))
                color_map = {
                    "Very Low Risk": "#28a745",
                    "Low Risk": "#17a2b8",
                    "Medium Risk": "#ffc107",
                    "High Risk": "#fd7e14",
                    "Very High Risk": "#dc3545",
                }

                for group in risk_order:
                    mask = result_df["Risk_Group"] == group
                    ax4.scatter(
                        result_df[mask].index,
                        result_df[mask]["Risk_Score"],
                        c=color_map[group],
                        label=group,
                        alpha=0.6,
                        s=100,
                    )

                ax4.axhline(
                    y=0.5,
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                )
                ax4.set_xlabel(
                    "Patient Index",
                    fontsize=11,
                    fontweight="bold",
                )
                ax4.set_ylabel(
                    "Risk Score",
                    fontsize=11,
                    fontweight="bold",
                )
                ax4.set_title(
                    "Individual Patient Risk Scores",
                    fontsize=13,
                    fontweight="bold",
                    pad=15,
                )
                ax4.legend(loc="upper right", fontsize=8)
                ax4.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig4)

            # --------- 위험군별 환자 리스트 ---------
            st.markdown("### ⚠️ Patient Lists by Risk Group")

            subtab1, subtab2, subtab3 = st.tabs(
                [
                    "🔴 High Risk Patients",
                    "🟡 Medium Risk Patients",
                    "🟢 Low Risk Patients",
                ]
            )

            def format_patient_table(df_sub: pd.DataFrame) -> pd.DataFrame:
                table = df_sub[
                    ["Patient_ID", "Risk_Score", "Survival_Rate", "Risk_Group"]
                ].copy()
                table = table.sort_values("Risk_Score", ascending=False)
                table["Risk_Score"] = table["Risk_Score"].apply(
                    lambda x: f"{x:.3f}"
                )
                table["Survival_Rate"] = table["Survival_Rate"].apply(
                    lambda x: f"{x:.1f}%"
                )
                return table

            # 🔴 High Risk (High + Very High)
            with subtab1:
                high_df = result_df[
                    result_df["Risk_Group"].isin(
                        ["High Risk", "Very High Risk"]
                    )
                ]
                if high_df.empty:
                    st.info("현재 High / Very High Risk 환자가 없습니다.")
                else:
                    st.dataframe(
                        format_patient_table(high_df),
                        use_container_width=True,
                        hide_index=True,
                    )

            # 🟡 Medium Risk
            with subtab2:
                med_df = result_df[result_df["Risk_Group"] == "Medium Risk"]
                if med_df.empty:
                    st.info("현재 Medium Risk 환자가 없습니다.")
                else:
                    st.dataframe(
                        format_patient_table(med_df),
                        use_container_width=True,
                        hide_index=True,
                    )

            # 🟢 Low Risk (Low + Very Low)
            with subtab3:
                low_df = result_df[
                    result_df["Risk_Group"].isin(
                        ["Low Risk", "Very Low Risk"]
                    )
                ]
                if low_df.empty:
                    st.info("현재 Low / Very Low Risk 환자가 없습니다.")
                else:
                    st.dataframe(
                        format_patient_table(low_df),
                        use_container_width=True,
                        hide_index=True,
                    )

            # --------- 결과 다운로드 ---------
            st.markdown("### 💾 Download Results")

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Prediction Results (CSV)",
                data=csv_bytes,
                file_name=f"MM_Risk_Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Please check your CSV file format and try again.")

# =======================================================
# 탭 2: Clinical Interpretation
# =======================================================
with tab2:
    render_clinical_tab()
