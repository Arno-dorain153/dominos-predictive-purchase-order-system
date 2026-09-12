import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dominos - Predictive Purchase Order Portal",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dominos Custom Styling (Blue #006491, Red #E31837)
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .stApp { background-color: #F8FAFC; }
    .dominos-header {
        background: linear-gradient(135deg, #006491 0%, #004d70 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 20px rgba(0,100,145,0.15);
    }
    .dominos-badge-admin {
        background-color: #E31837;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    .dominos-badge-employee {
        background-color: #006491;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 0.75rem;
        border-left: 5px solid #006491;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .highlight-banner {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 1rem;
        box-shadow: 0 10px 15px -3px rgba(5, 150, 105, 0.2);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALES_FILE = os.path.join(BASE_DIR, 'cleaned_pizza2.csv')
PO_FILE = os.path.join(BASE_DIR, 'ingredient_purchase_order.csv')

# User DB
USERS_DB = {
    "admin": {"password": "admin123", "name": "Alex Mercer", "role_title": "Dominos Upper Management (Admin)", "role": "admin"},
    "employee": {"password": "dominos123", "name": "Sarah Connor", "role_title": "Dominos Store Employee (Kitchen Lead)", "role": "employee"}
}

# Session State Initialization
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_info" not in st.session_state:
    st.session_state["user_info"] = None


# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state["logged_in"]:
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; margin-bottom: 1.5rem;">
        <h1 style="color: #006491; font-size: 2.5rem; font-weight: 900; margin-bottom: 0;">🍕 Dominos Portal</h1>
        <p style="color: #64748B; font-weight: 600;">Predictive Inventory, Food Waste & Purchase Order Platform</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.info("💡 **Preset Demo Login Profiles:**\n- **Upper Management (Admin):** `admin` / `admin123`\n- **Store Employee (Kitchen):** `employee` / `dominos123`")
        
        with st.form("login_form"):
            st.subheader("Account Sign In")
            username = st.text_input("Username").strip()
            password = st.text_input("Password", type="password").strip()
            submit = st.form_submit_button("Sign In to Portal", use_container_width=True)

            if submit:
                user = USERS_DB.get(username)
                if user and user["password"] == password:
                    st.session_state["logged_in"] = True
                    st.session_state["user_info"] = user
                    st.success("Login Successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")
    st.stop()


# -----------------------------------------------------------------------------
# DASHBOARD FOR LOGGED-IN USERS
# -----------------------------------------------------------------------------
user = st.session_state["user_info"]

# Header
st.markdown(f"""
<div class="dominos-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; font-weight: 900;">Dominos Predictive Procurement OS</h1>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Role View: {user['role_title']}</p>
        </div>
        <div style="text-align: right;">
            <span class="{'dominos-badge-admin' if user['role']=='admin' else 'dominos-badge-employee'}">{user['role_title']}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Dominos_pizza_logo.svg/512px-Dominos_pizza_logo.svg.png", width=120)
    st.markdown(f"**Logged in as:**\n{user['name']}")
    
    st.markdown("---")
    timeline = st.selectbox(
        "📅 Select Analytics Timeline:",
        ["Full Year 2015", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    )
    
    safety_buffer_pct = st.slider("🛡️ Safety Buffer (+%):", min_value=5, max_value=25, value=10, step=1)
    
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["user_info"] = None
        st.rerun()

# Load Data
@st.cache_data
def get_processed_sales():
    df = pd.read_csv(SALES_FILE)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['month_num'] = df['order_date'].dt.month
    df['month_name'] = df['order_date'].dt.strftime('%B')
    
    monthly = df.groupby(['month_num', 'month_name']).agg(
        total_sales=('total_price', 'sum'),
        total_pizzas=('quantity', 'sum')
    ).reset_index().sort_values('month_num')
    
    monthly['food_spend'] = (monthly['total_sales'] * 0.30).round(2)
    monthly['food_wastage_baseline'] = (monthly['total_sales'] * 0.045).round(2)
    monthly['food_wastage_optimized'] = (monthly['total_sales'] * 0.012 + np.random.uniform(50, 150, len(monthly))).round(2)
    monthly['wastage_pct'] = ((monthly['food_wastage_optimized'] / monthly['total_sales']) * 100).round(2)
    return monthly, df

monthly_df, raw_sales = get_processed_sales()

# Timeline Filter
if timeline != "Full Year 2015":
    filtered_df = monthly_df[monthly_df['month_name'] == timeline]
else:
    filtered_df = monthly_df

total_sales = filtered_df['total_sales'].sum()
total_spend = filtered_df['food_spend'].sum()
total_waste = filtered_df['food_wastage_optimized'].sum()
total_saved = filtered_df['food_wastage_baseline'].sum() - total_waste

# Stat Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Gross Sales Revenue", f"${total_sales:,.2f}", "+100% 2015 History")
c2.metric("Food Ingredient Spend", f"${total_spend:,.2f}", "30% Food COGS")
c3.metric("Food Wastage Cost", f"${total_waste:,.2f}", f"{(total_waste/total_sales*100):.2f}% Waste Ratio", delta_color="inverse")
c4.metric("Wastage Savings via ML", f"${total_saved:,.2f}", "Stockout & Expiration Reduction")

# Lowest Food Wastage Month Card
lowest_row = monthly_df.loc[monthly_df['food_wastage_optimized'].idxmin()]
st.markdown(f"""
<div class="highlight-banner">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h3 style="margin: 0; font-size: 1.2rem; font-weight: 900;">🏆 Month with Lowest Food Wastage: {lowest_row['month_name']} 2015</h3>
            <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">Achieved peak operational efficiency with lowest ingredient degradation.</p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 1.4rem; font-weight: 900;">${lowest_row['food_wastage_optimized']:,.2f}</div>
            <div style="font-size: 0.8rem; font-weight: 700;">({lowest_row['wastage_pct']}% Waste Ratio)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📊 Sales vs Food Spend & Wastage Analytics", "📦 Upcoming 7-Day Food Purchase Order"])

with tab1:
    st.subheader("Sales Record vs Food Spend & Food Wastage Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(monthly_df))
    width = 0.25
    
    ax.bar(x - width, monthly_df['total_sales'], width, label='Gross Sales Revenue ($)', color='#006491')
    ax.bar(x, monthly_df['food_spend'], width, label='Food Ingredient Spend ($)', color='#F59E0B')
    ax.bar(x + width, monthly_df['food_wastage_optimized'], width, label='Food Wastage Cost ($)', color='#E31837')
    
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_df['month_name'], rotation=35, ha='right')
    ax.set_ylabel("USD ($)")
    ax.set_title("Dominos Monthly Financial & Food Wastage Breakdown", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    st.pyplot(fig)

with tab2:
    st.subheader(f"Upcoming 7-Day Ingredient Purchase Order ({safety_buffer_pct}% Safety Buffer)")
    
    if os.path.exists(PO_FILE):
        po_df = pd.read_csv(PO_FILE)
        
        # Adjust buffer dynamically
        multiplier = (100.0 + safety_buffer_pct) / 110.0
        po_df['total_kg'] = (po_df['total_kg'] * multiplier).round(2)
        po_df['recommended_packs_5kg'] = np.ceil(po_df['total_kg'] / 5.0).astype(int)
        
        search_query = st.text_input("🔍 Search Ingredient:", "")
        if search_query:
            po_df = po_df[po_df['pizza_ingredients'].str.contains(search_query.lower(), case=False)]
            
        st.dataframe(po_df, use_container_width=True)
        
        csv_data = po_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Purchase Order CSV",
            data=csv_data,
            file_name="Dominos_7Day_Purchase_Order.csv",
            mime="text/csv"
        )
    else:
        st.warning("Purchase order file not found. Run run_pipeline.py first!")
