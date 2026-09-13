"""Streamlit retention dashboard. Talks to the FastAPI service over HTTP only.

Run with::

    streamlit run dashboard/app.py

Set ``CHURN_API_URL`` to point at a non-default API host (default http://127.0.0.1:8000).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from churn.features import (  # noqa: E402  (import after sys.path tweak)
    ALL_FEATURES,
    CATEGORY_VALUES,
    DATA_PATH,
    ID_COLUMN,
    INTERNET_ADDON_FEATURES,
    TARGET,
    clean,
    load_raw,
)

DEFAULT_API_URL = os.environ.get("CHURN_API_URL", "http://127.0.0.1:8000")

# --- palette: one categorical hue for magnitude, reserved status colours for tiers ---
SERIES_BLUE = "#2a78d6"
TIER_COLOR = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
TIER_TINT = {"Low": "rgba(12,163,12,0.14)", "Medium": "rgba(250,178,25,0.18)", "High": "rgba(208,59,59,0.14)"}
TIER_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
GRID = "rgba(128,128,128,0.18)"

st.set_page_config(page_title="Churn Retention Dashboard", page_icon="📉", layout="wide")


# ----------------------------------------------------------------------------- API helpers
def api_get(base_url: str, path: str) -> dict[str, Any]:
    response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=60)
    if response.status_code == 422:
        raise ValueError(f"API rejected the payload: {response.json().get('detail')}")
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_info(base_url: str) -> dict[str, Any]:
    return api_get(base_url, "/model/info")


@st.cache_data(show_spinner=False)
def load_customers() -> pd.DataFrame:
    return clean(load_raw(DATA_PATH))


@st.cache_data(show_spinner="Scoring the customer base through the API…")
def score_customer_base(base_url: str) -> pd.DataFrame:
    """Score every customer in the dataset via /predict/batch and attach revenue at risk."""
    customers = load_customers()
    payload = {"customers": customers[ALL_FEATURES].to_dict("records")}
    body = api_post(base_url, "/predict/batch", payload)
    preds = pd.DataFrame(body["predictions"])
    scored = pd.concat([customers.reset_index(drop=True), preds], axis=1)
    scored["already_churned"] = scored[TARGET] == 1
    return scored.sort_values("monthly_revenue_at_risk", ascending=False).reset_index(drop=True)


def normalize_customer(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Make the add-on fields consistent with phone/internet service so the API accepts them."""
    customer = dict(raw)
    notes: list[str] = []
    if customer["PhoneService"] == "No" and customer["MultipleLines"] != "No phone service":
        customer["MultipleLines"] = "No phone service"
        notes.append("MultipleLines set to 'No phone service'")
    if customer["PhoneService"] == "Yes" and customer["MultipleLines"] == "No phone service":
        customer["MultipleLines"] = "No"
        notes.append("MultipleLines set to 'No'")
    for addon in INTERNET_ADDON_FEATURES:
        if customer["InternetService"] == "No" and customer[addon] != "No internet service":
            customer[addon] = "No internet service"
            notes.append(f"{addon} set to 'No internet service'")
        elif customer["InternetService"] != "No" and customer[addon] == "No internet service":
            customer[addon] = "No"
            notes.append(f"{addon} set to 'No'")
    return customer, notes


def retention_levers(customer: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Plausible retention offers, each expressed as a modified copy of the customer."""
    levers: list[tuple[str, dict[str, Any]]] = []
    if customer["Contract"] == "Month-to-month":
        levers.append(("Move to one-year contract", {**customer, "Contract": "One year"}))
    if customer["Contract"] != "Two year":
        levers.append(("Move to two-year contract", {**customer, "Contract": "Two year"}))
    if customer["InternetService"] != "No":
        if customer["TechSupport"] == "No":
            levers.append(("Add tech support", {**customer, "TechSupport": "Yes"}))
        if customer["OnlineSecurity"] == "No":
            levers.append(("Add online security", {**customer, "OnlineSecurity": "Yes"}))
    if "automatic" not in customer["PaymentMethod"]:
        levers.append(
            ("Switch to automatic bank transfer", {**customer, "PaymentMethod": "Bank transfer (automatic)"})
        )
    if customer["MonthlyCharges"] >= 20:
        discounted = round(customer["MonthlyCharges"] * 0.9, 2)
        levers.append((f"10% loyalty discount (${discounted:.2f}/mo)", {**customer, "MonthlyCharges": discounted}))
    return levers


# ----------------------------------------------------------------------------- chart helpers
def tidy(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(gridcolor=GRID, gridwidth=1, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, showline=False)
    return fig


def gauge(probability: float, threshold: float, tier: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": "Churn probability", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickwidth": 1, "tickcolor": GRID},
                "bar": {"color": TIER_COLOR[tier], "thickness": 0.35},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": TIER_TINT["Low"]},
                    {"range": [30, 60], "color": TIER_TINT["Medium"]},
                    {"range": [60, 100], "color": TIER_TINT["High"]},
                ],
                "threshold": {
                    "line": {"color": "#898781", "width": 2},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=24, r=24, t=48, b=8), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def hbar(frame: pd.DataFrame, x: str, y: str, title: str, hover: str, text_fmt: str, height: int = 420) -> go.Figure:
    """Single-series horizontal bar: thin, rounded data-end, value at the tip."""
    data = frame.iloc[::-1]  # plotly draws bottom-up; keep the largest on top
    fig = go.Figure(
        go.Bar(
            x=data[x],
            y=data[y],
            orientation="h",
            marker=dict(color=SERIES_BLUE, cornerradius=4),
            text=data[x].map(text_fmt.format),
            textposition="outside",
            cliponaxis=False,
            hovertemplate=hover + "<extra></extra>",
            customdata=data.to_dict("records"),
        )
    )
    fig.update_layout(title=dict(text=title, font=dict(size=15)), bargap=0.45)
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showgrid=False, type="category")
    return tidy(fig, height)


def money(value: float) -> str:
    return f"${value:,.0f}"


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📉 Churn Retention")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL, help="Where the FastAPI service is running")
    try:
        health = api_get(api_url, "/health")
        st.success(f"API online · model: **{health['model_name'].replace('_', ' ')}**")
        api_ok = True
    except requests.RequestException as exc:  # connection refused, timeout, 5xx
        st.error("API unreachable. Start it with `make api` (or `uvicorn api.main:app --port 8000`).")
        st.caption(str(exc))
        api_ok = False
    st.markdown(
        """
        **How it works**

        1. `train.py` fits a scikit-learn pipeline on the IBM Telco churn dataset.
        2. FastAPI serves it at `/predict` and `/predict/batch`.
        3. This dashboard calls the API and ranks customers by
           **churn probability × monthly bill** = expected revenue at risk.
        """
    )

if not api_ok:
    st.stop()

info = fetch_model_info(api_url)
threshold = float(info["threshold"])

st.title("Customer Churn Predictor & Retention Dashboard")
tab_predict, tab_board, tab_model = st.tabs(["🔮 Predict a customer", "💸 Retention risk board", "📊 Model performance"])


# ----------------------------------------------------------------------------- tab 1: single prediction
with tab_predict:
    st.caption("Change any attribute and press **Predict** to query the API. Then compare retention offers.")
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Account**")
            tenure = st.slider("Tenure (months)", 0, 72, 3)
            monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 85.0, step=0.5)
            contract = st.selectbox("Contract", CATEGORY_VALUES["Contract"])
            payment = st.selectbox("Payment method", CATEGORY_VALUES["PaymentMethod"])
            paperless = st.selectbox("Paperless billing", CATEGORY_VALUES["PaperlessBilling"])
            total_charges = st.number_input(
                "Total charges ($) · 0 = estimate from tenure × monthly", min_value=0.0, value=0.0, step=10.0
            )
        with c2:
            st.markdown("**Services**")
            phone = st.selectbox("Phone service", CATEGORY_VALUES["PhoneService"])
            multiple_lines = st.selectbox("Multiple lines", CATEGORY_VALUES["MultipleLines"], index=1)
            internet = st.selectbox("Internet service", CATEGORY_VALUES["InternetService"], index=1)
            online_security = st.selectbox("Online security", CATEGORY_VALUES["OnlineSecurity"], index=1)
            online_backup = st.selectbox("Online backup", CATEGORY_VALUES["OnlineBackup"], index=1)
            device_protection = st.selectbox("Device protection", CATEGORY_VALUES["DeviceProtection"], index=1)
            tech_support = st.selectbox("Tech support", CATEGORY_VALUES["TechSupport"], index=1)
            streaming_tv = st.selectbox("Streaming TV", CATEGORY_VALUES["StreamingTV"])
            streaming_movies = st.selectbox("Streaming movies", CATEGORY_VALUES["StreamingMovies"])
        with c3:
            st.markdown("**Customer**")
            gender = st.selectbox("Gender", CATEGORY_VALUES["gender"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", CATEGORY_VALUES["Partner"], index=1)
            dependents = st.selectbox("Dependents", CATEGORY_VALUES["Dependents"], index=1)
        submitted = st.form_submit_button("Predict churn risk", type="primary", width="stretch")

    if submitted:
        raw_customer = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges) if total_charges > 0 else None,
        }
        customer, notes = normalize_customer(raw_customer)
        if notes:
            st.info("Adjusted for consistency with the selected services: " + "; ".join(notes))
        try:
            st.session_state["prediction"] = api_post(api_url, "/predict", customer)
            st.session_state["customer"] = customer
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Prediction failed: {exc}")

    prediction = st.session_state.get("prediction")
    customer = st.session_state.get("customer")
    if prediction and customer:
        tier = prediction["risk_tier"]
        p = prediction["churn_probability"]
        left, right = st.columns([1, 1.4])
        with left:
            st.plotly_chart(gauge(p, threshold, tier), width="stretch", config={"displayModeBar": False})
            st.caption(f"Grey marker = decision threshold ({threshold:.0%}). Bands: low < 30%, medium 30–60%, high ≥ 60%.")
        with right:
            m1, m2 = st.columns(2)
            m1.metric("Risk tier", f"{TIER_ICON[tier]} {tier}")
            m2.metric("Flagged for outreach", "Yes" if prediction["will_churn"] else "No")
            m3, m4 = st.columns(2)
            m3.metric("Monthly revenue at risk", f"${prediction['monthly_revenue_at_risk']:,.2f}")
            m4.metric("Annual revenue at risk", f"${prediction['annual_revenue_at_risk']:,.2f}")
            st.caption(
                f"Revenue at risk = churn probability ({p:.1%}) × monthly charges "
                f"(${prediction['monthly_charges']:,.2f})."
            )

        st.subheader("Retention levers: what would change this customer's risk?")
        levers = retention_levers(customer)
        if not levers:
            st.write("This customer is already on the lowest-risk plan configuration.")
        else:
            try:
                batch = api_post(api_url, "/predict/batch", {"customers": [c for _, c in levers]})
            except (requests.RequestException, ValueError) as exc:
                st.error(f"Could not score retention levers: {exc}")
            else:
                rows = []
                for (name, variant), pred in zip(levers, batch["predictions"]):
                    new_p = pred["churn_probability"]
                    rows.append(
                        {
                            "Retention lever": name,
                            "New churn probability": new_p,
                            "Change (pts)": round((new_p - p) * 100, 1),
                            "New tier": f"{TIER_ICON[pred['risk_tier']]} {pred['risk_tier']}",
                            "Monthly revenue protected": round(
                                prediction["monthly_revenue_at_risk"] - pred["monthly_revenue_at_risk"], 2
                            ),
                        }
                    )
                lever_df = pd.DataFrame(rows).sort_values("Monthly revenue protected", ascending=False)
                lc1, lc2 = st.columns([1.2, 1])
                with lc1:
                    st.dataframe(
                        lever_df,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "New churn probability": st.column_config.ProgressColumn(
                                "New churn probability", format="percent", min_value=0, max_value=1
                            ),
                            "Change (pts)": st.column_config.NumberColumn(format="%+.1f"),
                            "Monthly revenue protected": st.column_config.NumberColumn(format="$%.2f"),
                        },
                    )
                with lc2:
                    st.plotly_chart(
                        hbar(
                            lever_df,
                            x="Monthly revenue protected",
                            y="Retention lever",
                            title="Expected monthly revenue protected per lever",
                            hover="%{y}<br>Protects $%{x:,.2f} per month",
                            text_fmt="${:,.2f}",
                            height=max(220, 60 + 48 * len(lever_df)),
                        ),
                        width="stretch",
                        config={"displayModeBar": False},
                    )
                st.caption(
                    "Each lever re-scores the same customer with one attribute changed. "
                    "This is the model's estimate of the offer's effect, not a causal guarantee."
                )
    else:
        st.info("Fill in the form and press **Predict churn risk** to see the score, risk tier and revenue at risk.")


# ----------------------------------------------------------------------------- tab 2: retention board
with tab_board:
    try:
        scored = score_customer_base(api_url)
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Could not score the customer base: {exc}")
        st.stop()

    f1, f2, f3, f4, f5 = st.columns([1.1, 1.3, 1.3, 1.1, 0.8])
    include_churned = f1.toggle("Include customers who already left", value=False)
    tiers = f2.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"])
    contracts = f3.multiselect("Contract", CATEGORY_VALUES["Contract"], default=CATEGORY_VALUES["Contract"])
    min_bill = f4.slider("Min monthly bill ($)", 0, 120, 0, step=5)
    top_n = f5.number_input("Top N", min_value=5, max_value=200, value=25, step=5)

    view = scored[
        (scored["risk_tier"].isin(tiers))
        & (scored["Contract"].isin(contracts))
        & (scored["MonthlyCharges"] >= min_bill)
    ]
    if not include_churned:
        view = view[~view["already_churned"]]
    view = view.reset_index(drop=True)

    if view.empty:
        st.warning("No customers match the current filters.")
        st.stop()

    total_monthly = float(view["monthly_revenue_at_risk"].sum())
    top = view.head(int(top_n))
    top_share = float(top["monthly_revenue_at_risk"].sum()) / total_monthly if total_monthly else 0.0

    st.markdown(
        f"<div style='font-size:0.95rem;color:#898781'>Expected monthly revenue at risk (filtered customers)</div>"
        f"<div style='font-size:3rem;font-weight:600;line-height:1.1'>{money(total_monthly)}</div>",
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers in view", f"{len(view):,}")
    k2.metric("Flagged for outreach", f"{int(view['will_churn'].sum()):,}", help=f"churn probability ≥ {threshold:.0%}")
    k3.metric("Annualised revenue at risk", money(total_monthly * 12))
    k4.metric(f"Share held by top {int(top_n)}", f"{top_share:.0%}", help="How concentrated the risk is")

    ch1, ch2 = st.columns([1.3, 1])
    with ch1:
        top_chart = top.head(15).assign(label=lambda d: d[ID_COLUMN] + "  ·  " + d["Contract"])
        st.plotly_chart(
            hbar(
                top_chart,
                x="monthly_revenue_at_risk",
                y="label",
                title="Top 15 accounts by expected monthly revenue at risk",
                hover="%{y}<br>$%{x:,.2f}/mo at risk · P(churn) %{customdata.churn_probability:.0%} · bill $%{customdata.MonthlyCharges:,.2f}",
                text_fmt="${:,.0f}",
                height=520,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    with ch2:
        by_contract = (
            view.groupby("Contract", observed=True)["monthly_revenue_at_risk"]
            .sum()
            .reindex(CATEGORY_VALUES["Contract"])
            .dropna()
            .reset_index()
        )
        fig = go.Figure(
            go.Bar(
                x=by_contract["Contract"],
                y=by_contract["monthly_revenue_at_risk"],
                marker=dict(color=SERIES_BLUE, cornerradius=4),
                text=by_contract["monthly_revenue_at_risk"].map(money),
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>$%{y:,.0f}/mo at risk<extra></extra>",
                width=0.5,
            )
        )
        fig.update_layout(title=dict(text="Monthly revenue at risk by contract type", font=dict(size=15)))
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(tidy(fig, 520), width="stretch", config={"displayModeBar": False})

    st.subheader(f"High-value retention risk list · top {int(top_n)}")
    table = top[
        [
            ID_COLUMN,
            "risk_tier",
            "churn_probability",
            "MonthlyCharges",
            "monthly_revenue_at_risk",
            "annual_revenue_at_risk",
            "Contract",
            "tenure",
            "InternetService",
            "PaymentMethod",
            "already_churned",
        ]
    ].assign(risk_tier=lambda d: d["risk_tier"].map(lambda t: f"{TIER_ICON[t]} {t}"))
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            ID_COLUMN: "Customer",
            "risk_tier": "Risk tier",
            "churn_probability": st.column_config.ProgressColumn("Churn probability", format="percent", min_value=0, max_value=1),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly bill", format="$%.2f"),
            "monthly_revenue_at_risk": st.column_config.NumberColumn("Monthly $ at risk", format="$%.2f"),
            "annual_revenue_at_risk": st.column_config.NumberColumn("Annual $ at risk", format="$%.0f"),
            "tenure": st.column_config.NumberColumn("Tenure (mo)"),
            "InternetService": "Internet",
            "PaymentMethod": "Payment",
            "already_churned": st.column_config.CheckboxColumn("Already left"),
        },
    )
    st.download_button(
        "Download full ranked list (CSV)",
        data=view.to_csv(index=False).encode(),
        file_name="retention_risk_list.csv",
        mime="text/csv",
    )
    st.caption(
        "Ranking = churn probability × monthly charges, i.e. the expected monthly revenue lost if nobody acts. "
        "Scores for customers in the training split are in-sample; see the README for the caveat."
    )


# ----------------------------------------------------------------------------- tab 3: model performance
with tab_model:
    holdout = info["holdout"]
    impact = info["business_impact"]
    st.caption(
        f"Best of {len(info['candidates'])} candidates by 5-fold CV ROC-AUC: **{info['best_model'].replace('_', ' ')}**. "
        f"Trained on {info['dataset']['train_rows']:,} customers, evaluated on a held-out {info['dataset']['holdout_rows']:,}. "
        f"Threshold {threshold:.3f} chosen on out-of-fold predictions to maximise F1."
    )
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("ROC-AUC", f"{holdout['roc_auc']:.3f}")
    p2.metric("Accuracy", f"{holdout['accuracy']:.1%}")
    p3.metric("Recall (churners caught)", f"{holdout['recall']:.1%}")
    p4.metric("Precision", f"{holdout['precision']:.1%}")
    p5.metric("F1", f"{holdout['f1']:.3f}")

    st.subheader("Business impact on the holdout set")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Churning revenue captured", f"{impact['revenue_capture_rate']:.1%}",
              help="Share of monthly revenue from actual churners that the model flagged in time")
    b2.metric("Customers flagged", f"{impact['flag_rate']:.1%}",
              help=f"{impact['customers_flagged']:,} of {impact['holdout_customers']:,} holdout customers")
    b3.metric("Churning revenue caught", money(impact["monthly_revenue_of_churners_flagged"]),
              help=f"Monthly revenue of flagged customers who really churned, "
                   f"out of {money(impact['monthly_revenue_of_churners'])} that churned")
    b4.metric("Revenue of all flagged", money(impact["monthly_revenue_of_all_flagged"]),
              help=f"Monthly revenue of every flagged customer. {impact['flag_revenue_precision']:.1%} of it "
                   f"belongs to actual churners; the rest is retention spend on customers who would have stayed")

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        imp = pd.DataFrame(info["feature_importance"]).head(12)
        st.plotly_chart(
            hbar(
                imp,
                x="importance",
                y="feature",
                title="Permutation importance (drop in ROC-AUC when shuffled)",
                hover="%{y}<br>ΔAUC %{x:.4f}",
                text_fmt="{:.3f}",
                height=460,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    with col_b:
        st.markdown("**Candidate models**")
        cand = pd.DataFrame(
            [
                {
                    "Model": name.replace("_", " "),
                    "CV ROC-AUC": f"{c['cv_roc_auc_mean']:.4f} ± {c['cv_roc_auc_std']:.4f}",
                    "Holdout ROC-AUC": c["holdout_at_0.5"]["roc_auc"],
                    "Holdout accuracy": c["holdout_at_0.5"]["accuracy"],
                }
                for name, c in info["candidates"].items()
            ]
        )
        st.dataframe(cand, hide_index=True, width="stretch")

        st.markdown(f"**Confusion matrix at threshold {threshold:.3f}**")
        cm = info["confusion_matrix"]
        st.dataframe(
            pd.DataFrame(
                {"Predicted: stays": [cm["tn"], cm["fn"]], "Predicted: churns": [cm["fp"], cm["tp"]]},
                index=["Actually stayed", "Actually churned"],
            ),
            width="stretch",
        )
        st.caption(
            f"Trained {info['trained_at'][:19].replace('T', ' ')} UTC · scikit-learn {info['environment']['scikit_learn']}"
        )
