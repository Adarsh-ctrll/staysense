import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="NYC Airbnb Room Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 24px;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, rgba(80,80,90,.12), rgba(180,180,190,.08));
            border: 1px solid rgba(128,128,128,.22);
        }

        .eyebrow {
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
            opacity: .68;
            margin-bottom: .5rem;
        }

        .hero h1 {
            font-size: clamp(2.2rem, 5vw, 4.4rem);
            line-height: .98;
            margin: 0;
            letter-spacing: -.045em;
        }

        .hero p {
            max-width: 780px;
            font-size: 1.05rem;
            opacity: .78;
            margin-top: 1rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 750;
            margin: .5rem 0 1rem;
        }

        .prediction-card {
            padding: 1.5rem;
            border-radius: 22px;
            border: 1px solid rgba(128,128,128,.22);
            background: rgba(128,128,128,.07);
            margin-bottom: 1rem;
        }

        .prediction-label {
            font-size: .75rem;
            text-transform: uppercase;
            letter-spacing: .14em;
            opacity: .65;
            font-weight: 700;
        }

        .prediction-name {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -.035em;
            margin-top: .25rem;
        }

        .confidence {
            font-size: 1.1rem;
            margin-top: .3rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid rgba(128,128,128,.18);
            background: rgba(128,128,128,.055);
        }

        .metric-label {
            font-size: .72rem;
            text-transform: uppercase;
            letter-spacing: .1em;
            opacity: .62;
        }

        .metric-value {
            font-size: 1.45rem;
            font-weight: 750;
            margin-top: .2rem;
        }

        .info-box {
            padding: 1rem 1.1rem;
            border-radius: 15px;
            border: 1px solid rgba(128,128,128,.18);
            background: rgba(128,128,128,.055);
            margin-top: .8rem;
        }

        .footer {
            text-align: center;
            opacity: .55;
            font-size: .8rem;
            margin-top: 2.5rem;
        }

        div[data-testid="stProgress"] > div > div {
            border-radius: 999px;
        }

        button[kind="primary"] {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model_pipeline.pkl")


try:
    model = load_model()
    model_error = None
except Exception as e:
    model = None
    model_error = str(e)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def display_name(label):
    mapping = {
        "Entire home/apt": "Entire Home / Apartment",
        "Private room": "Private Room",
        "Shared room": "Shared Room",
    }
    return mapping.get(str(label), str(label))


def probability_table(classes, probabilities):
    rows = []
    for cls, prob in zip(classes, probabilities):
        rows.append(
            {
                "Room Type": display_name(cls),
                "Probability": float(prob),
            }
        )
    return pd.DataFrame(rows).sort_values("Probability", ascending=False)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏙️ NYC Airbnb")
    st.caption("Room Type Predictor")

    st.divider()

    st.markdown("### About this app")
    st.write(
        "Enter listing characteristics and the trained scikit-learn "
        "pipeline predicts the most likely Airbnb room type."
    )

    st.markdown("### Model")
    st.write("Random Forest + preprocessing pipeline")

    st.markdown("### Classes")
    st.write("• Entire home / apartment")
    st.write("• Private room")
    st.write("• Shared room")

    st.divider()
    st.caption("FastAPI version can use the same model_pipeline.pkl.")

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">NYC Airbnb · Machine Learning</div>
        <h1>What kind of stay<br>is this listing?</h1>
        <p>
            Explore how price, location, reviews, availability and host
            activity influence the predicted Airbnb room type.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_error:
    st.error(
        "Could not load model_pipeline.pkl. Make sure it is in the same "
        "folder as app.py."
    )
    st.code(model_error)
    st.stop()

# ------------------------------------------------------------
# EXAMPLE
# ------------------------------------------------------------
EXAMPLE = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "price": 150.0,
    "minimum_nights": 2,
    "number_of_reviews": 20,
    "reviews_per_month": 1.5,
    "calculated_host_listings_count": 2,
    "availability_365": 200,
    "neighbourhood_group": "Brooklyn",
    "neighbourhood": "Williamsburg",
}

if "listing_values" not in st.session_state:
    st.session_state["listing_values"] = EXAMPLE.copy()

if st.button("✨ Load example listing", use_container_width=False):
    st.session_state["listing_values"] = EXAMPLE.copy()
    st.rerun()

v = st.session_state["listing_values"]

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="section-title">Listing details</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**📍 Location**")

        c1, c2 = st.columns(2)
        with c1:
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=float(v["latitude"]),
                format="%.6f",
            )
        with c2:
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=float(v["longitude"]),
                format="%.6f",
            )

        c1, c2 = st.columns(2)
        with c1:
            borough = st.selectbox(
                "Borough",
                ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
                index=["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"].index(
                    v["neighbourhood_group"]
                ) if v["neighbourhood_group"] in
                ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"] else 0,
            )
        with c2:
            neighbourhood = st.text_input(
                "Neighbourhood",
                value=v["neighbourhood"],
                placeholder="e.g. Williamsburg",
            )

    with st.container(border=True):
        st.markdown("**💵 Pricing & stay**")

        c1, c2 = st.columns(2)
        with c1:
            price = st.number_input(
                "Price per night (USD)",
                min_value=1.0,
                value=float(v["price"]),
                step=5.0,
            )
        with c2:
            minimum_nights = st.number_input(
                "Minimum nights",
                min_value=1,
                max_value=365,
                value=int(v["minimum_nights"]),
                step=1,
            )

        availability = st.slider(
            "Availability per year",
            min_value=0,
            max_value=365,
            value=int(v["availability_365"]),
        )
        st.caption(f"{availability} / 365 days available")

    with st.container(border=True):
        st.markdown("**⭐ Reviews & host**")

        c1, c2 = st.columns(2)
        with c1:
            number_of_reviews = st.number_input(
                "Total reviews",
                min_value=0,
                value=int(v["number_of_reviews"]),
                step=1,
            )
        with c2:
            reviews_per_month = st.number_input(
                "Reviews per month",
                min_value=0.0,
                value=float(v["reviews_per_month"]),
                step=0.1,
            )

        calculated_host_listings_count = st.number_input(
            "Listings by this host",
            min_value=0,
            value=int(v["calculated_host_listings_count"]),
            step=1,
        )

    predict_clicked = st.button(
        "🚀 Predict room type",
        type="primary",
        use_container_width=True,
    )

# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------
with right:
    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

    if not predict_clicked:
        st.info(
            "Enter the listing details and click **Predict room type** "
            "to see the model's prediction and confidence."
        )

        st.markdown(
            """
            <div class="info-box">
                <b>What you'll get</b><br><br>
                • Predicted room type<br>
                • Probability for every class<br>
                • Model confidence<br>
                • Class-by-class comparison
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        row = pd.DataFrame(
            [{
                "latitude": latitude,
                "longitude": longitude,
                "price": price,
                "minimum_nights": minimum_nights,
                "number_of_reviews": number_of_reviews,
                "reviews_per_month": reviews_per_month,
                "calculated_host_listings_count": calculated_host_listings_count,
                "availability_365": availability,
                "neighbourhood_group": borough,
                "neighbourhood": neighbourhood,
            }]
        )

        try:
            prediction = model.predict(row)[0]
            probabilities = model.predict_proba(row)[0]
            classes = model.classes_

            confidence = float(max(probabilities))
            result_df = probability_table(classes, probabilities)

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">Most likely room type</div>
                    <div class="prediction-name">{display_name(prediction)}</div>
                    <div class="confidence">
                        Model confidence: <b>{confidence:.1%}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Prediction</div>
                        <div class="metric-value">{display_name(prediction)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{confidence:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown("### Class probabilities")

            for _, r in result_df.iterrows():
                prob = r["Probability"]
                st.write(f"**{r['Room Type']}** — {prob:.1%}")
                st.progress(min(max(prob, 0.0), 1.0))

            # Minority-class warning
            shared = result_df[
                result_df["Room Type"] == "Shared Room"
            ]["Probability"]

            if not shared.empty:
                shared_prob = float(shared.iloc[0])

                if shared_prob < 0.20:
                    st.caption(
                        "ℹ️ Shared rooms are a minority class in the training data, "
                        "so the model may be more conservative when predicting them."
                    )

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Built with Streamlit · scikit-learn · pandas · Random Forest
        <br>
        NYC Airbnb Room Type Classification
    </div>
    """,
    unsafe_allow_html=True,
)