
import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_USERNAME = "LeaMiles"
MODEL_REPO = f"{HF_USERNAME}/tourism-wellness-model"

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="\U0001F334", layout="wide")


@st.cache_resource
def load_model():
    # Cached so the model is only downloaded/deserialized once per app session, not on every click
    model_path = hf_hub_download(repo_id=MODEL_REPO, repo_type="model", filename="model.joblib")
    return joblib.load(model_path)


model = load_model()


@st.cache_resource
def load_feature_importance():
    # Reuses the SAME CSV train.py uploaded next to the model, so the app never has to recompute anything
    path = hf_hub_download(repo_id=MODEL_REPO, repo_type="model", filename="feature_importance.csv")
    return pd.read_csv(path, index_col=0)


feature_importance_df = load_feature_importance()

# Inject custom CSS so the primary ("Predict") button matches GitHub's green,
# making the main action visually obvious at a glance
st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #2ea44f;
        color: white;
        border: 1px solid #2ea44f;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #2c974b;
        border: 1px solid #2c974b;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Hero banner -----------------------------------------------------------
# Hand-drawn inline SVG (sun / waves / palm tree) -- no external image service,
# so nothing can break due to network issues or discontinued placeholder APIs.
HERO_SVG = """
<div style="border-radius:14px;overflow:hidden;margin-bottom:1.2rem;">
  <svg viewBox="0 0 800 180" xmlns="http://www.w3.org/2000/svg" width="100%" height="180">
    <defs>
      <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#2f9e91"/>
        <stop offset="100%" stop-color="#79d1c4"/>
      </linearGradient>
    </defs>
    <rect width="800" height="180" fill="url(#sky)"/>
    <circle cx="670" cy="55" r="34" fill="#ffd76a"/>
    <path d="M0 140 Q100 110 200 140 T400 140 T600 140 T800 140 V180 H0 Z" fill="#1f7f74" opacity="0.6"/>
    <path d="M0 155 Q100 130 200 155 T400 155 T600 155 T800 155 V180 H0 Z" fill="#166b62"/>
    <g transform="translate(90,60)">
      <rect x="-4" y="30" width="8" height="70" fill="#7a4a25"/>
      <path d="M0 30 C -40 10 -60 -20 -70 -35 C -40 -30 -10 -10 0 20 Z" fill="#2fae63"/>
      <path d="M0 30 C 40 10 60 -20 70 -35 C 40 -30 10 -10 0 20 Z" fill="#2fae63"/>
      <path d="M0 30 C -30 20 -55 0 -60 -25 C -30 -15 -10 5 0 25 Z" fill="#39c473"/>
      <path d="M0 30 C 30 20 55 0 60 -25 C 30 -15 10 5 0 25 Z" fill="#39c473"/>
    </g>
    <text x="230" y="100" font-family="Arial, sans-serif" font-size="30" fill="white" font-weight="bold">
      Wellness Tourism Package Predictor
    </text>
  </svg>
</div>
"""
st.markdown(HERO_SVG, unsafe_allow_html=True)

st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# --- Package-tier badge icons ----------------------------------------------
# Small inline SVGs, one per ProductPitched tier, shown next to that selectbox
TIER_BADGES = {
    "Basic": """<svg width="34" height="34" viewBox="0 0 24 24"><rect x="4" y="8" width="16" height="12" rx="2" fill="#9aa5b1"/><rect x="8" y="5" width="8" height="4" rx="1" fill="#7b8794"/></svg>""",
    "Standard": """<svg width="34" height="34" viewBox="0 0 24 24"><path d="M12 3 L12 21 M4 9 C4 5 20 5 20 9 Z" stroke="#2b8bd6" stroke-width="2" fill="#5fb3e8"/></svg>""",
    "Deluxe": """<svg width="34" height="34" viewBox="0 0 24 24"><circle cx="12" cy="9" r="5" fill="#ffb648"/><rect x="3" y="16" width="18" height="4" rx="2" fill="#2f9e91"/></svg>""",
    "Super Deluxe": """<svg width="34" height="34" viewBox="0 0 24 24"><path d="M12 2 C 5 6 5 14 5 14 C5 20 19 20 19 14 C19 14 19 6 12 2Z" fill="#8e5fd6"/><circle cx="12" cy="10" r="2" fill="#ffffff"/></svg>""",
    "King": """<svg width="34" height="34" viewBox="0 0 24 24"><path d="M3 18 L4 9 L8 13 L12 6 L16 13 L20 9 L21 18 Z" fill="#f2c14e" stroke="#c9971f" stroke-width="0.5"/></svg>""",
}

# Default value for every field, used both on first load and when "Clear Fields" is clicked
defaults = {
    "age": 35,
    "type_of_contact": "Self Enquiry",
    "city_tier": 1,
    "duration_of_pitch": 15,
    "occupation": "Salaried",
    "gender": "Male",
    "num_person_visiting": 3,
    "num_followups": 3,
    "product_pitched": "Basic",
    "preferred_property_star": 3,
    "marital_status": "Single",
    "num_trips": 3,
    "passport": "No",
    "pitch_satisfaction_score": 3,
    "own_car": "No",
    "num_children_visiting": 0,
    "designation": "Executive",
    "monthly_income": 22000,
}

# Seed session_state with defaults the first time the app loads
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_fields():
    # Overwriting session_state and letting Streamlit rerun is what actually clears the widgets on screen
    for key, value in defaults.items():
        st.session_state[key] = value


col1, col2 = st.columns(2)

with col1:
    st.number_input("Age", min_value=18, max_value=100, key="age")
    st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"], key="type_of_contact")
    st.selectbox("City Tier", [1, 2, 3], key="city_tier")
    st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, key="duration_of_pitch")
    st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"], key="occupation")
    st.selectbox("Gender", ["Male", "Female"], key="gender")
    st.number_input("Number of Persons Visiting", min_value=1, max_value=10, key="num_person_visiting")
    st.number_input("Number of Followups", min_value=0, max_value=10, key="num_followups")

    # Product Pitched selectbox + its live tier badge, side by side
    pp_col1, pp_col2 = st.columns([4, 1])
    with pp_col1:
        st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"], key="product_pitched")
    with pp_col2:
        st.markdown(f"<div style='margin-top:1.8rem'>{TIER_BADGES[st.session_state.product_pitched]}</div>", unsafe_allow_html=True)

with col2:
    st.selectbox("Preferred Property Star", [3, 4, 5], key="preferred_property_star")
    st.selectbox("Marital Status", ["Single", "Married", "Divorced"], key="marital_status")
    st.number_input("Number of Trips per Year", min_value=0, max_value=20, key="num_trips")
    st.selectbox("Has Passport", ["No", "Yes"], key="passport")
    st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], key="pitch_satisfaction_score")
    st.selectbox("Owns a Car", ["No", "Yes"], key="own_car")
    st.number_input("Number of Children Visiting", min_value=0, max_value=5, key="num_children_visiting")
    st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"], key="designation")
    st.number_input("Monthly Income", min_value=0, max_value=200000, key="monthly_income")

# Two side-by-side action buttons: a prominent green Predict, and a plain Clear Fields
button_col1, button_col2 = st.columns(2)

with button_col1:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

with button_col2:
    st.button("Clear Fields", on_click=reset_fields, use_container_width=True)

if predict_clicked:
    # Read current values straight from session_state -- these are exactly what the widgets show on screen
    input_df = pd.DataFrame([{
        "Age": st.session_state.age,
        "TypeofContact": st.session_state.type_of_contact,
        "CityTier": st.session_state.city_tier,
        "DurationOfPitch": st.session_state.duration_of_pitch,
        "Occupation": st.session_state.occupation,
        "Gender": st.session_state.gender,
        "NumberOfPersonVisiting": st.session_state.num_person_visiting,
        "NumberOfFollowups": st.session_state.num_followups,
        "ProductPitched": st.session_state.product_pitched,
        "PreferredPropertyStar": st.session_state.preferred_property_star,
        "MaritalStatus": st.session_state.marital_status,
        "NumberOfTrips": st.session_state.num_trips,
        "Passport": 1 if st.session_state.passport == "Yes" else 0,
        "PitchSatisfactionScore": st.session_state.pitch_satisfaction_score,
        "OwnCar": 1 if st.session_state.own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": st.session_state.num_children_visiting,
        "Designation": st.session_state.designation,
        "MonthlyIncome": st.session_state.monthly_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # A large, unmistakable visual card: green + thumbs-up for Yes, red + thumbs-down for No
    if prediction == 1:
        st.markdown(
            f"""
            <div style="background-color:#e6f4ea;border:2px solid #34a853;border-radius:12px;
                        padding:24px;text-align:center;margin-top:14px;">
                <div style="font-size:54px;line-height:1;">\U0001F44D</div>
                <div style="font-size:24px;font-weight:700;color:#1e7e34;margin-top:6px;">
                    Likely to Purchase
                </div>
                <div style="font-size:15px;color:#3c4043;margin-top:4px;">
                    Predicted probability: {probability:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#fce8e6;border:2px solid #d93025;border-radius:12px;
                        padding:24px;text-align:center;margin-top:14px;">
                <div style="font-size:54px;line-height:1;">\U0001F44E</div>
                <div style="font-size:24px;font-weight:700;color:#a50e0e;margin-top:6px;">
                    Unlikely to Purchase
                </div>
                <div style="font-size:15px;color:#3c4043;margin-top:4px;">
                    Predicted probability: {probability:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Model insights: always visible, not just after clicking Predict -------
with st.expander("\U0001F4CA See what the model bases its predictions on"):
    st.write("Relative importance of the top features driving this model's predictions:")
    st.bar_chart(feature_importance_df["importance"])
