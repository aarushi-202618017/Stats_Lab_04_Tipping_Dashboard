import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(
    page_title="Restaurant Tipping Behavior Analytics", layout="wide"
)


# Helper function to eliminate 'e' notation across p-values
def format_p(val):
  if val < 0.0001:
    return "< 0.0001"
  return f"{val:.4f}"


@st.cache_data
def load_data():
  return pd.read_csv("data/tips.csv")


df = load_data()

st.title("Restaurant Tipping Behavior & Statistical Modeling Dashboard")

tab1, tab2, tab3 = st.tabs([
    "Data Exploration",
    "Hypothesis Testing Lab",
    "Live Prediction & Diagnostics",
])

# ---------------------------------------------------------
# TAB 1: DATA EXPLORATION
# ---------------------------------------------------------
with tab1:
  st.sidebar.header("Data Filters")
  bill_range = st.sidebar.slider(
      "Total Bill ($)",
      float(df["total_bill"].min()),
      float(df["total_bill"].max()),
      (float(df["total_bill"].min()), float(df["total_bill"].max())),
  )
  day_filter = st.sidebar.multiselect(
      "Day of Week", df["day"].unique(), default=df["day"].unique()
  )
  time_filter = st.sidebar.multiselect(
      "Time of Day", df["time"].unique(), default=df["time"].unique()
  )

  filtered_df = df[
      (df["total_bill"] >= bill_range[0])
      & (df["total_bill"] <= bill_range[1])
      & (df["day"].isin(day_filter))
      & (df["time"].isin(time_filter))
  ]

  st.subheader("Descriptive Metrics")
  num_cols = ["total_bill", "tip", "size"]
  desc_df = pd.DataFrame({
      "Mean": filtered_df[num_cols].mean(),
      "Median": filtered_df[num_cols].median(),
      "Std Dev": filtered_df[num_cols].std(),
      "IQR": filtered_df[num_cols].apply(stats.iqr),
      "Skewness": filtered_df[num_cols].skew(),
      "Kurtosis": filtered_df[num_cols].kurtosis(),
  })
  st.dataframe(desc_df.style.format("{:.2f}"))

  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Bivariate Analysis: Bill vs. Tip")
    fig_scatter = px.scatter(
        filtered_df,
        x="total_bill",
        y="tip",
        color="smoker",
        size="size",
        hover_data=["day", "time"],
        trendline="ols",
        title="Total Bill vs. Tip Amount",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

  with col2:
    st.subheader("Correlation Heatmap")
    corr = filtered_df[num_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: HYPOTHESIS TESTING LAB
# ---------------------------------------------------------
with tab2:
  st.subheader("Hypothesis Test 1: Compare 2 Independent Groups")
  cat_group = st.selectbox(
      "Select Grouping Variable (Binary)", ["sex", "smoker", "time"], index=1
  )
  num_target = st.selectbox(
      "Select Target Variable", ["tip", "total_bill"], index=0
  )

  groups = filtered_df[cat_group].unique()
  if len(groups) == 2:
    g1 = filtered_df[filtered_df[cat_group] == groups[0]][num_target]
    g2 = filtered_df[filtered_df[cat_group] == groups[1]][num_target]

    stat_s1, p_s1 = stats.shapiro(g1)
    stat_s2, p_s2 = stats.shapiro(g2)
    stat_lev, p_lev = stats.levene(g1, g2)

    is_normal = (p_s1 > 0.05) and (p_s2 > 0.05)

    st.write(
        f"**Normality Check (Shapiro-Wilk):** `{groups[0]}` p={format_p(p_s1)},"
        f" `{groups[1]}` p={format_p(p_s2)}"
    )
    st.write(
        "**Homogeneity of Variance Check (Levene's Test):**"
        f" p={format_p(p_lev)}"
    )

    if is_normal:
      stat, p_val = stats.ttest_ind(g1, g2, equal_var=(p_lev > 0.05))
      test_name = "Two-Sample t-Test"
    else:
      stat, p_val = stats.mannwhitneyu(g1, g2)
      test_name = "Mann-Whitney U Test (Non-Parametric)"

    st.write(f"**Executed Test:** {test_name}")
    st.write(
        f"**Test Statistic:** {stat:.4f} | **p-value:** {format_p(p_val)}"
    )

    if p_val < 0.05:
      st.error(
          f"Conclusion: Reject H0 at α = 0.05. Statistically significant"
          f" difference in {num_target} between {groups[0]} and {groups[1]}."
      )
    else:
      st.success(
          f"Conclusion: Fail to Reject H0 at α = 0.05. No significant difference"
          f" detected in {num_target} between groups."
      )
  else:
    st.warning(
        "Group comparison requires exactly two categories. Adjust your filters"
        " in the sidebar."
    )

  st.markdown("---")
  st.subheader("Hypothesis Test 2: One-Way ANOVA across Days")

  available_days = filtered_df["day"].nunique()

  if available_days < 2:
    st.warning(
        "One-Way ANOVA requires comparison across at least 2 days. Please"
        " select more than one day in the sidebar filter."
    )
  else:
    day_groups = [
        group[num_target].values for name, group in filtered_df.groupby("day")
    ]
    f_stat, p_val_anova = stats.f_oneway(*day_groups)

    st.write(f"**Test:** One-Way ANOVA across days for `{num_target}`")
    st.write(
        f"**F-Statistic:** {f_stat:.4f} | **p-value:** {format_p(p_val_anova)}"
    )

    if p_val_anova < 0.05:
      st.error(
          "Conclusion: Reject H0 at α = 0.05. Mean target value differs"
          " significantly across days."
      )
    else:
      st.success(
          "Conclusion: Fail to Reject H0 at α = 0.05. No statistically"
          " significant difference across days."
      )

# ---------------------------------------------------------
# TAB 3: LIVE PREDICTION & DIAGNOSTICS
# ---------------------------------------------------------
with tab3:
  st.subheader("Statistical Modeling (OLS Regression)")

  formula = "tip ~ total_bill + size + C(sex) + C(smoker) + C(day) + C(time)"
  model = smf.ols(formula, data=df).fit()

  col_left, col_right = st.columns([1, 1])

  with col_left:
    st.write("**Interactive Tip Predictor**")
    input_bill = st.slider(
        "Total Bill ($)",
        float(df["total_bill"].min()),
        float(df["total_bill"].max()),
        20.0,
    )
    input_size = st.slider(
        "Party Size", int(df["size"].min()), int(df["size"].max()), 2
    )
    input_sex = st.selectbox("Gender", df["sex"].unique())
    input_smoker = st.selectbox("Smoker Section", df["smoker"].unique())
    input_day = st.selectbox("Day of Week", df["day"].unique())
    input_time = st.selectbox("Time of Day", df["time"].unique())

    user_data = pd.DataFrame({
        "total_bill": [input_bill],
        "size": [input_size],
        "sex": [input_sex],
        "smoker": [input_smoker],
        "day": [input_day],
        "time": [input_time],
    })

    pred = model.get_prediction(user_data).summary_frame(alpha=0.05)

    st.metric("Predicted Tip Amount", f"${pred['mean'].values[0]:,.2f}")
    st.write(
        "**95% Confidence Interval (Mean Tip):**"
        f" [${pred['mean_ci_lower'].values[0]:,.2f},"
        f" ${pred['mean_ci_upper'].values[0]:,.2f}]"
    )
    st.write(
        "**95% Prediction Interval (Individual Tip):**"
        f" [${pred['obs_ci_lower'].values[0]:,.2f},"
        f" ${pred['obs_ci_upper'].values[0]:,.2f}]"
    )

  with col_right:
    st.write("**Model Fit Summary**")
    st.write(f"**R² Score:** {model.rsquared:.4f}")
    st.write(f"**Adjusted R²:** {model.rsquared_adj:.4f}")
    st.write(f"**F-Statistic p-value:** {format_p(model.f_pvalue)}")
    st.write(
        "**Residual Normality (Jarque-Bera p-value):**"
        f" {format_p(stats.jarque_bera(model.resid).pvalue)}"
    )

  st.markdown("---")
  st.subheader("Gauss-Markov Residual Diagnostics & Multicollinearity")

  diag_col1, diag_col2, diag_col3 = st.columns(3)

  with diag_col1:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(model.fittedvalues, model.resid, alpha=0.5, color="darkblue")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Fitted")
    st.pyplot(fig)

  with diag_col2:
    fig, ax = plt.subplots(figsize=(4, 3))
    sm.qqplot(model.resid, line="45", ax=ax)
    ax.set_title("Normal Q-Q Plot")
    st.pyplot(fig)

  with diag_col3:
    st.write("**VIF Multicollinearity Check**")
    X_num = df[["total_bill", "size"]].dropna()
    X_num = sm.add_constant(X_num)
    vif_data = pd.DataFrame({
        "Variable": X_num.columns,
        "VIF": [
            variance_inflation_factor(X_num.values, i)
            for i in range(X_num.shape[1])
        ],
    })
    st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}))
