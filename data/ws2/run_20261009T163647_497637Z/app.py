
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
import html
from pathlib import Path

st.set_page_config(page_title="BoE Workstream 2 Analytics Dashboard", layout="wide")

st.title("Bank of England — WS2 Analytics")
st.markdown("WS1 data & WS2 analysis.")
st.sidebar.header("Filters")

@st.cache_data
def load_web_dashboard_data():
    data_path = Path(__file__).resolve().parent / "LLMSignals.csv"
    if not data_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(data_path)
    for col in ['directness_score', 'qa_dissonance_hybrid']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df = load_web_dashboard_data()

if df.empty:
    st.error("LLMSignals.csv not found or is empty. Please run the pipeline export first.")
    st.stop()

# Single clean instance of the filters (duplication removed)
banks_list = ["Included-bank pooled average"] + sorted(df['bank'].dropna().unique().tolist())
selected_bank = st.sidebar.selectbox("Select Target Banking Entity", banks_list)

topics_list = ["ALL Topics"] + sorted(df['question_topic_label'].dropna().unique().tolist())
selected_topic = st.sidebar.selectbox("Select Classified Conversation Topic", topics_list)

filtered_df = df.copy()
if selected_bank != "Included-bank pooled average":
    filtered_df = filtered_df[filtered_df['bank'] == selected_bank]
if selected_topic != "ALL Topics":
    filtered_df = filtered_df[filtered_df['question_topic_label'] == selected_topic]

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# 1. KPI Calculation (Current vs Previous Quarter Deltas)
quarters = sorted(filtered_df['period'].dropna().unique())
current_q_data = filtered_df[filtered_df['period'] == quarters[-1]] if len(quarters) > 0 else filtered_df
prev_q_data = filtered_df[filtered_df['period'] == quarters[-2]] if len(quarters) > 1 else pd.DataFrame()

curr_dir = current_q_data['directness_score'].mean()
prev_dir = prev_q_data['directness_score'].mean() if not prev_q_data.empty else np.nan
dir_delta = f"{curr_dir - prev_dir:.2f}" if pd.notna(curr_dir) and pd.notna(prev_dir) else None

curr_diss = current_q_data['qa_dissonance_hybrid'].mean()
prev_diss = prev_q_data['qa_dissonance_hybrid'].mean() if not prev_q_data.empty else np.nan
diss_delta = f"{curr_diss - prev_diss:.2f}" if pd.notna(curr_diss) and pd.notna(prev_diss) else None

m1, m2, m3 = st.columns(3)
m1.metric("Scoped exchanges", len(filtered_df))
m2.metric(f"Mean Directness ({quarters[-1] if quarters else 'Overall'})", f"{curr_dir:.2f}" if pd.notna(curr_dir) else "N/A", dir_delta)
m3.metric(f"Mean Dissonance ({quarters[-1] if quarters else 'Overall'})", f"{curr_diss:.2f}" if pd.notna(curr_diss) else "N/A", diss_delta, delta_color="inverse")

st.caption(f'LLM coverage: {int(filtered_df.llm_status.eq("ok").sum())}/{len(filtered_df)} accepted assessments. *The pooled average is exchange-weighted across all included banks.*')

st.markdown("---")

# Initialize the Streamlit Tabs
tab1, tab2 = st.tabs(["Macro Performance", "Micro Vulnerability"])

# Tab 1: The Timeline
with tab1:
    if selected_bank == "Included-bank pooled average":
        timeline_data = filtered_df.groupby(['period']).agg(
            mean_dissonance=('qa_dissonance_hybrid', 'mean')
        ).reset_index().sort_values('period')
        timeline_data['bank'] = 'Included-bank pooled average'
    else:
        timeline_data = filtered_df.groupby(['period', 'bank']).agg(
            mean_dissonance=('qa_dissonance_hybrid', 'mean')
        ).reset_index().sort_values('period')

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    if timeline_data.mean_dissonance.isna().all():
        st.info('No accepted hybrid scores in this selection.')
    else:
        for bank_name, group in timeline_data.groupby('bank'):
            fig.add_trace(
                go.Scatter(
                    x=group['period'], y=group['mean_dissonance'], name=f"{bank_name} Hybrid Dissonance",
                    mode='lines+markers', line=dict(width=3.5)
                )
            )
        fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=70, b=10), height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: The Heatmap
with tab2:
    st.subheader("Conversation Topic Directness")

    # Injects exactly 5 blank text lines for spacing
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

    pivot_table = filtered_df.pivot_table(index='question_topic_label', columns='period', values='directness_score', aggfunc='mean')

    if not pivot_table.empty:
        pivot_table['row_mean'] = pivot_table.mean(axis=1)
        pivot_table = pivot_table.sort_values('row_mean', ascending=True).drop(columns=['row_mean'])

        # Restored RdBu palette and full-width auto aspect
        fig_heat = px.imshow(pivot_table, color_continuous_scale='RdBu', zmin=0.0, zmax=1.0, aspect="auto")
        fig_heat.update_layout(margin=dict(l=10, r=10, t=70, b=10), height=max(400, len(pivot_table)*30))
        fig_heat.update_yaxes(title_text="")
        fig_heat.update_traces(hovertemplate="<b>Period:</b> %{x}<br><b>Topic:</b> %{y}<br><b>Directness:</b> %{z:.2f}<extra></extra>")

        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Insufficient topic data to render heatmap across periods.")

st.markdown("---")

# 4. Keyword Highlights - Placed at the bottom, full width
st.subheader("Keyword Highlights & Evidence")

accepted_df = filtered_df[filtered_df['llm_status'] == 'ok'].reset_index(drop=True)

if len(accepted_df) > 0:
    selected_row_idx = st.slider("Select exchange", 0, len(accepted_df)-1, 0) if len(accepted_df) > 1 else 0
    row = accepted_df.iloc[selected_row_idx]
    neg_tokens = {"compression", "hedged", "vulnerability", "provisions", "delinquency", "evasive", "risk"}
    pos_tokens = {"resilient", "capital", "framework", "complete", "satisfied", "direct", "buffer"}

    def markup_words(text):
        if pd.isna(text): return "N/A"
        buffer = []
        for w in str(text).split(" "):
            cw = re.sub(r'[^\w\s]', '', w).lower()
            w = html.escape(w)
            if cw in neg_tokens:
                buffer.append(f'<span style="background-color:rgba(231,76,60,0.3); padding:2px; border-radius:3px; font-weight:bold;">{w}</span>')
            elif cw in pos_tokens:
                buffer.append(f'<span style="background-color:rgba(39,174,96,0.3); padding:2px; border-radius:3px; font-weight:bold;">{w}</span>')
            else:
                buffer.append(f" {w} ")
        return "".join(buffer)

    st.markdown(f"**Target Entity:** `{row.get('bank', 'Unknown')}` | **Reporting Quarter:** `{row.get('period', 'Unknown')}` | **Review Priority:** `{row.get('human_review_priority', 'ROUTINE').upper()}`")

    col_q, col_a = st.columns(2)
    with col_q:
        st.info("**Analyst Question Quote:**")
        st.markdown(f"{markup_words(row.get('llm_question_quote', ''))}", unsafe_allow_html=True)
    with col_a:
        st.success("**Management Answer Quote:**")
        st.markdown(f"{markup_words(row.get('llm_answer_quote', ''))}", unsafe_allow_html=True)

    st.markdown(f"**LLM Reasoning:** *{row.get('llm_reason', 'No reasoning provided.')}*")
else:
    st.info("No LLM-assessed exchanges available for the current filter selection.")
