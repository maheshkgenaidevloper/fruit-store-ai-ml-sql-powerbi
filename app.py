
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "fruit_store_sales.csv"
MODEL_FILE = ROOT / "models" / "units_sold_model.joblib"

st.set_page_config(
    page_title="Fruit Store Intelligence",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Fruit Store Intelligence")
st.caption("AI/ML | Data Science | SQL | Power BI")

# Load data
if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info(
        "Create the data folder and add "
        "fruit_store_sales.csv inside it."
    )
    st.stop()

df = pd.read_csv(DATA_FILE, parse_dates=["date"])

required = [
    "date", "fruit", "category", "unit_price",
    "units_sold", "promotion", "store"
]

missing = [col for col in required if col not in df.columns]
if missing:
    st.error(f"Missing CSV columns: {missing}")
    st.stop()

# Clean data and calculate metrics
df = df.dropna(subset=required).copy()
df["revenue"] = df["unit_price"] * df["units_sold"]
df["month_start"] = (
    df["date"].dt.to_period("M").dt.to_timestamp()
)

# Demonstration assumption: 28% margin, not actual profit.
df["estimated_profit"] = df["revenue"] * 0.28

# Sidebar filters
st.sidebar.header("Filter Sales")

fruits = sorted(df["fruit"].unique())
stores = sorted(df["store"].unique())

selected_fruits = st.sidebar.multiselect(
    "Select fruit",
    fruits,
    default=fruits
)

selected_stores = st.sidebar.multiselect(
    "Select store",
    stores,
    default=stores
)

filtered = df[
    df["fruit"].isin(selected_fruits)
    & df["store"].isin(selected_stores)
]

if filtered.empty:
    st.warning("Select at least one fruit and one store.")
    st.stop()

# KPI cards
total_revenue = filtered["revenue"].sum()
total_units = int(filtered["units_sold"].sum())
estimated_profit = filtered["estimated_profit"].sum()
average_price = (
    total_revenue / total_units if total_units else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Revenue", f"${total_revenue:,.2f}")
c2.metric("Units Sold", f"{total_units:,}")
c3.metric("Estimated Profit", f"${estimated_profit:,.2f}")
c4.metric("Revenue per Unit", f"${average_price:,.2f}")

# Revenue by fruit
st.subheader("Fruit Sales Performance")

fruit_sales = (
    filtered.groupby("fruit", as_index=False)
    .agg(
        revenue=("revenue", "sum"),
        units_sold=("units_sold", "sum")
    )
)

left, right = st.columns(2)

with left:
    fig = px.bar(
        fruit_sales,
        x="fruit",
        y="revenue",
        title="Revenue by Fruit",
        text_auto=".2s"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        fruit_sales,
        x="fruit",
        y="units_sold",
        title="Units Sold by Fruit"
    )
    st.plotly_chart(fig, use_container_width=True)

# Monthly trend
st.subheader("Monthly Revenue Trend")

monthly = (
    filtered.groupby("month_start", as_index=False)
    .agg(revenue=("revenue", "sum"))
)

fig = px.line(
    monthly,
    x="month_start",
    y="revenue",
    markers=True,
    title="Revenue Over Time"
)

st.plotly_chart(fig, use_container_width=True)

# Store analysis and promotion analysis
left, right = st.columns(2)

with left:
    store_sales = (
        filtered.groupby("store", as_index=False)
        .agg(revenue=("revenue", "sum"))
    )

    fig = px.pie(
        store_sales,
        names="store",
        values="revenue",
        title="Revenue Share by Store"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    promo = (
        filtered.groupby("promotion", as_index=False)
        .agg(avg_units=("units_sold", "mean"))
    )

    promo["promotion_status"] = promo["promotion"].map({
        0: "No Promotion",
        1: "Promotion"
    })

    fig = px.bar(
        promo,
        x="promotion_status",
        y="avg_units",
        title="Average Units by Promotion"
    )
    st.plotly_chart(fig, use_container_width=True)

# View and download data
st.subheader("Sales Records")

st.dataframe(
    filtered.sort_values("date", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download Filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="fruit_store_filtered.csv",
    mime="text/csv"
)

# Machine learning prediction
st.divider()
st.header("🤖 Predict Fruit Sales")

if not MODEL_FILE.exists():
    st.warning(
        "The model has not been trained yet. "
        "Run: python train_model.py"
    )
else:
    model_data = joblib.load(MODEL_FILE)
    model = model_data["pipeline"]

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        fruit = col1.selectbox("Fruit", fruits)

        categories = sorted(
            df.loc[df["fruit"] == fruit, "category"].unique()
        )
        category = col2.selectbox("Category", categories)
        store = col3.selectbox("Store", stores)

        col4, col5, col6 = st.columns(3)

        default_price = float(
            df.loc[df["fruit"] == fruit, "unit_price"].median()
        )

        price = col4.number_input(
            "Unit Price ($)",
            min_value=0.1,
            max_value=100.0,
            value=default_price,
            step=0.1
        )

        promotion = col5.selectbox(
            "Promotion Active?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        month = col6.selectbox(
            "Month",
            list(range(1, 13)),
            format_func=lambda x: pd.Timestamp(
                2025, x, 1
            ).strftime("%B")
        )

        submitted = st.form_submit_button("Predict Units Sold")

    if submitted:
        input_data = pd.DataFrame([{
            "fruit": fruit,
            "category": category,
            "unit_price": price,
            "promotion": promotion,
            "month": month,
            "store": store
        }])

        prediction = max(
            0, round(float(model.predict(input_data)[0]))
        )

        st.success(
            f"Estimated units sold: {prediction:,}"
        )

        st.caption(
            "This estimate uses a small synthetic dataset. "
            "Validate it using real sales history before "
            "making inventory decisions."
        )
