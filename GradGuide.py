import streamlit as st
import pandas as pd
import joblib
import requests
import plotly.express as px
import plotly.graph_objects as go
#import base64 #for adding the background
from datetime import datetime, timedelta
import numpy as np


# Page config
st.set_page_config(
    page_title="GradGuide - Smart Career Planner",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("career_path_model.pkl")

model = load_model()

# --- Load Static Data at the Start ---
# This data can be moved to an external file like universities.json for easier updates
UNIVERSITY_DATA = {
    "USA": [
        "Massachusetts Institute of Technology", "Stanford University", "Harvard University",
        "California Institute of Technology", "University of California Berkeley",
        "Carnegie Mellon University", "Georgia Institute of Technology", "University of Illinois",
        "University of Michigan", "University of Washington", "Cornell University",
        "University of Texas at Austin", "Princeton University", "UCLA", "Columbia University"
    ],
    "UK": [
        "University of Cambridge", "University of Oxford", "Imperial College London",
        "University College London", "King's College London", "University of Edinburgh",
        "University of Manchester", "London School of Economics", "University of Warwick",
        "University of Bristol", "University of Glasgow", "Durham University",
        "University of Sheffield", "University of Nottingham", "University of Southampton"
    ],
    "Germany": [
        "Technical University of Munich", "ETH Zurich", "University of Heidelberg",
        "Ludwig Maximilian University", "Humboldt University Berlin", "RWTH Aachen University",
        "University of Freiburg", "University of Göttingen", "Technical University of Berlin",
        "University of Hamburg", "University of Stuttgart", "Karlsruhe Institute of Technology",
        "University of Cologne", "University of Münster", "University of Würzburg"
    ],
    "India": [
        "Indian Institute of Technology Delhi", "Indian Institute of Technology Bombay",
        "Indian Institute of Technology Madras", "Indian Institute of Technology Kanpur",
        "Indian Institute of Technology Kharagpur", "Indian Institute of Science Bangalore",
        "National Institute of Technology Trichy", "Delhi Technological University",
        "Birla Institute of Technology", "Vellore Institute of Technology",
        "Indian Institute of Technology Roorkee", "BITS Pilani", "Anna University",
        "Jadavpur University", "Indian Institute of Technology Guwahati"
    ],
    "Canada": [
        "University of Toronto", "University of British Columbia", "McGill University",
        "University of Alberta", "University of Waterloo", "McMaster University",
        "University of Montreal", "University of Calgary", "Queen's University",
        "Simon Fraser University", "University of Ottawa", "Western University",
        "University of Victoria", "Concordia University", "Carleton University"
    ],
    "Australia": [
        "Australian National University", "University of Melbourne", "University of Sydney",
        "University of Queensland", "University of New South Wales", "Monash University",
        "University of Western Australia", "University of Adelaide", "Macquarie University",
        "Queensland University of Technology", "University of Technology Sydney",
        "Griffith University", "Deakin University", "Curtin University", "RMIT University"
    ]
}

# Extended university data with more comprehensive mappings
EXTENDED_UNIVERSITY_DATA = {
    "United States": UNIVERSITY_DATA["USA"],
    "USA": UNIVERSITY_DATA["USA"],
    "US": UNIVERSITY_DATA["USA"],
    "United Kingdom": UNIVERSITY_DATA["UK"],
    "UK": UNIVERSITY_DATA["UK"],
    "Britain": UNIVERSITY_DATA["UK"],
    "Germany": UNIVERSITY_DATA["Germany"],
    "Deutschland": UNIVERSITY_DATA["Germany"],
    "India": UNIVERSITY_DATA["India"],
    "Bharat": UNIVERSITY_DATA["India"],
    "Canada": UNIVERSITY_DATA["Canada"],
    "Australia": UNIVERSITY_DATA["Australia"],
    "AUS": UNIVERSITY_DATA["Australia"]
}


# --- Function to Fetch Data with Caching ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_university_data(country):
    """Fetches university data from API or fallback, returns a DataFrame."""
    try:
        url = f"http://universities.hipolabs.com/search?country={country}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()
        if not data:
            st.info(f"Live search returned no results for {country}. Showing a cached list.")
            univ_names = UNIVERSITY_DATA.get(country, [])
        else:
            univ_names = [u["name"] for u in data[:15]]

    except requests.exceptions.RequestException:
        st.warning(f"Live university service is unavailable. Showing a cached list.")
        univ_names = UNIVERSITY_DATA.get(country, [])

    if not univ_names:
        return pd.DataFrame()  # Return empty DataFrame if no names found

    universities = []
    for univ in univ_names:
        universities.append({
            'University': univ,
            'QS Ranking': np.random.randint(1, 500),
            'Acceptance Rate': f"{np.random.randint(15, 85)}%",
            'Avg Fee (Lakhs)': np.random.randint(20, 100) if country != "India" else np.random.randint(5, 25),
            'Program Strength': np.random.choice(['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐'])
        })
    return pd.DataFrame(universities)


def get_cached_universities(country_input):
    """Get cached university data with flexible country matching."""
    # Try direct lookup first
    if country_input in EXTENDED_UNIVERSITY_DATA:
        return EXTENDED_UNIVERSITY_DATA[country_input]
    
    # Try case-insensitive lookup
    country_lower = country_input.lower()
    for key, value in EXTENDED_UNIVERSITY_DATA.items():
        if key.lower() == country_lower:
            return value
    
    # Try partial matching
    for key, value in EXTENDED_UNIVERSITY_DATA.items():
        if country_lower in key.lower() or key.lower() in country_lower:
            return value
    
    return []


def fetch_universities_with_fallback(country_input):
    """Fetch universities with robust fallback mechanism."""
    try:
        # Try API call first
        url = f"http://universities.hipolabs.com/search?country={country_input}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data:
            return data, "api"
        else:
            # API returned empty, use cached data
            cached_unis = get_cached_universities(country_input)
            if cached_unis:
                # Convert to API format for consistency
                cached_data = [{"name": uni, "web_pages": [f"https://{uni.lower().replace(' ', '')}.edu"]} for uni in cached_unis]
                return cached_data, "cached"
            else:
                return [], "none"
    
    except requests.exceptions.RequestException as e:
        # API failed, use cached data
        cached_unis = get_cached_universities(country_input)
        if cached_unis:
            cached_data = [{"name": uni, "web_pages": [f"https://{uni.lower().replace(' ', '')}.edu"]} for uni in cached_unis]
            return cached_data, "cached"
        else:
            return [], "error"


def search_university_by_name_with_fallback(university_name):
    """Search for specific university with fallback."""
    try:
        url = f"http://universities.hipolabs.com/search?name={university_name}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data:
            return data, "api"
        else:
            # Search in cached data
            found_unis = []
            search_term = university_name.lower()
            
            for country_unis in EXTENDED_UNIVERSITY_DATA.values():
                for uni in country_unis:
                    if search_term in uni.lower():
                        found_unis.append({
                            "name": uni,
                            "country": "Multiple countries available",
                            "state-province": "N/A",
                            "web_pages": [f"https://{uni.lower().replace(' ', '')}.edu"]
                        })
            
            return found_unis, "cached" if found_unis else "none"
    
    except requests.exceptions.RequestException:
        # API failed, search in cached data
        found_unis = []
        search_term = university_name.lower()
        
        for country_unis in EXTENDED_UNIVERSITY_DATA.values():
            for uni in country_unis:
                if search_term in uni.lower():
                    found_unis.append({
                        "name": uni,
                        "country": "Multiple countries available",
                        "state-province": "N/A",
                        "web_pages": [f"https://{uni.lower().replace(' ', '')}.edu"]
                    })
        
        return found_unis, "cached" if found_unis else "error"


# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: linear-gradient(45deg, #56CCF2, #2F80ED);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 GradGuide - Smart Career Planner</h1>
    <p>Your AI-powered companion for MS/MTech decision making</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🗺 Navigation")
step = st.sidebar.radio("Choose Your Journey", [
    "🎯 Career Prediction",
    "🏛 University Explorer",
    "💰 Financial Planner",
    "📊 Progress Tracker"
])

# Encoding dictionaries
career_goal_dict = {"Academia": 0, "Industry": 1, "Research": 2}
country_dict = {"USA": 0, "UK": 1, "Germany": 2, "India": 3, "Other": 4}

# ================ CAREER PREDICTION ================
if step == "🎯 Career Prediction":
    #set_background("https://images.unsplash.com/photo-1506784983877-45594efa4cbe")
    st.header("🎯 Find Your Perfect Path")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 Academic Profile")
        cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5, step=0.1, help="Your current CGPA")
        gre = st.number_input("GRE Score", 0, 340, 0, help="Enter 0 if not taken")
        toefl = st.number_input("TOEFL Score", 0, 120, 0, help="Enter 0 if not taken")
        gate_score = st.number_input("GATE Score", 0, 1000, 0, help="Enter 0 if not taken")

    with col2:
        st.subheader("🎯 Profile Strengths")
        sop = st.select_slider("SOP Quality", [1, 2, 3, 4, 5], 3, help="Statement of Purpose strength")
        lor = st.select_slider("LOR Quality", [1, 2, 3, 4, 5], 3, help="Letter of Recommendation strength")
        univ_rating = st.select_slider("Target University Rating", [1, 2, 3, 4, 5], 3)
        chance = st.slider("Self-assessed Admission Chance", 0.0, 1.0, 0.5, help="Your confidence level")

    col3, col4 = st.columns(2)
    with col3:
        research = st.selectbox("Research Experience", ["No", "Yes"])
        career_goal = st.selectbox("Career Goal", ["Industry", "Academia", "Research", "Entrepreneurship"])

    with col4:
        budget = st.number_input("Budget (INR Lakhs)", 1, 200, 25)
        pref_country = st.selectbox("Preferred Country", ["India", "USA", "UK", "Germany", "Canada", "Other"])

    if st.button("🚀 Get My Recommendation", type="primary"):
        # Process inputs
        research_val = 1 if research == "Yes" else 0
        career_goal_val = career_goal_dict.get(career_goal, 1)
        country_val = country_dict.get(pref_country, 4)

        input_data = pd.DataFrame([{
            "CGPA": cgpa, "GRE Score": gre, "TOEFL Score": toefl,
            "SOP": sop, "LOR ": lor, "University Rating": univ_rating,
            "Chance of Admit ": chance, "Research": research_val,
            "Budget (INR Lakhs)": budget, "Career Goal": career_goal_val,
            "GATE Score": gate_score, "Preferred Country": country_val
        }])

        prediction = model.predict(input_data)[0]
        path = "MS (Abroad)" if prediction == 1 else "MTech (India)"

        st.markdown(f"""
        <div class="success-box">
            <h2>🎯 Recommended Path: {path}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Preparation timeline
        st.subheader("📅 Suggested Preparation Timeline")

        if prediction == 1:  # MS
            timeline_data = {
                'Phase': ['Exam Prep', 'Applications', 'Interviews', 'Visa Process'],
                'Duration': [6, 3, 2, 2],
                'Priority': ['High', 'High', 'Medium', 'High']
            }
        else:  # MTech
            timeline_data = {
                'Phase': ['GATE Prep', 'College Research', 'Applications', 'Counseling'],
                'Duration': [8, 2, 1, 1],
                'Priority': ['High', 'Medium', 'High', 'Medium']
            }

        timeline_df = pd.DataFrame(timeline_data)
        fig = px.bar(timeline_df, x='Phase', y='Duration', color='Priority',
                     title="Preparation Timeline (Months)")
        st.plotly_chart(fig, use_container_width=True)

# ================ UNIVERSITY EXPLORER (API BASED WITH FALLBACK) ================
elif step == "🏛 University Explorer":
    st.header("🏛 Discover Your Dream Universities")

    country = st.text_input("Enter Country (e.g., United States, Canada, Germany)").strip()

    program = st.selectbox("Program Type", ["MS", "MTech", "PhD"])

    if st.button("Fetch Universities"):
        if country:
            with st.spinner(f"Searching for universities in {country}..."):
                data, source = fetch_universities_with_fallback(country)
                
                if data:
                    # Display source information
                    if source == "api":
                        st.success(f"✅ Found {len(data)} universities in {country} (Live data)")
                    elif source == "cached":
                        st.info(f"📋 Showing cached universities for {country} (API unavailable)")
                    
                    univ_list = [
                        {
                            "University Name": u["name"],
                            "Website": u["web_pages"][0] if u.get("web_pages") else "N/A"
                        }
                        for u in data[:20]  # Limit to top 20
                    ]
                    df = pd.DataFrame(univ_list)

                    st.success(f"🎓 Top {program} Universities in {country}")
                    st.dataframe(df, use_container_width=True)

                elif source == "none":
                    st.warning(f"⚠ No universities found for '{country}'. Please try another country name (e.g., 'United States', 'Canada', 'Germany').")
                else:  # source == "error"
                    st.error(f"⚠ Failed to fetch universities for '{country}' and no cached data available. Please try a different country name.")
        else:
            st.warning("⚠ Please enter a country name.")
    
    st.markdown("---")
    university_name = st.text_input("🎯 Or Search for a Specific University")

    if st.button("🔎 Search University by Name"):
        if university_name.strip():
            with st.spinner(f"Searching for '{university_name}'..."):
                results, source = search_university_by_name_with_fallback(university_name)
                
                if results:
                    if source == "api":
                        st.success(f"✅ Found {len(results)} universities matching '{university_name}' (Live data)")
                    elif source == "cached":
                        st.info(f"📋 Found {len(results)} universities matching '{university_name}' in cached data (API unavailable)")

                    for uni in results:
                        st.markdown(f"""
                            🎓 **{uni['name']}**  
                            🗺 Country: {uni.get('country', 'N/A')}  
                            🏛 State/Province: {uni.get('state-province', 'N/A')}  
                            🔗 [Website]({uni['web_pages'][0] if uni.get('web_pages') else '#'})
                            """)
                        st.divider()
                elif source == "none":
                    st.warning(f"⚠ No universities found matching '{university_name}'. Try a different search term.")
                else:  # source == "error"
                    st.error(f"❌ Search failed and no cached data found for '{university_name}'. Please try again later.")
        else:
            st.warning("⚠ Please enter a university name to search.")

# ================ FINANCIAL PLANNER ================
elif step == "💰 Financial Planner":
    st.header("💰 Smart Financial Planning")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💸 Cost Breakdown")
        tuition = st.number_input("Tuition Fee (INR Lakhs)", 1, 200, 40)
        living = st.number_input("Living Expenses (INR Lakhs)", 1, 100, 25)
        misc = st.number_input("Other Expenses (INR Lakhs)", 1, 50, 10)

    with col2:
        st.subheader("💡 Financial Aid")
        scholarship = st.slider("Expected Scholarship (%)", 0, 100, 20)
        family_support = st.number_input("Family Support (INR Lakhs)", 0, 150, 30)
        income_source = st.number_input("Another source of income/part-time",0,100000,300)
        #loan_needed = st.checkbox("Education Loan Required?")

    # Calculate costs
    total_cost = tuition + living + misc
    scholarship_amount = (tuition * scholarship) / 100
    part_time =income_source / 100
    net_cost = total_cost - scholarship_amount - family_support-part_time

    # Financial breakdown chart
    cost_data = {
        'Category': ['Tuition', 'Living', 'Others', 'Scholarship (Saved)', 'Family Support','part-time'],
        'Amount': [tuition, living, misc, -scholarship_amount, -family_support,-part_time],
        'Type': ['Expense', 'Expense', 'Expense', 'Savings', 'Support','income'],
    }

    fig = go.Figure()
    fig.add_trace(go.Waterfall(
        name="Financial Flow",
        orientation="v",
        x=cost_data['Category'],
        y=cost_data['Amount'],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title="Financial Breakdown (INR Lakhs)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Results
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("💰 Total Cost", f"₹{total_cost:.1f}L")
    with col4:
        st.metric("🎁 Total Savings", f"₹{scholarship_amount + family_support:.1f}L")
    with col5:
        if net_cost > 0:
            st.metric("📋 Loan Needed", f"₹{net_cost:.1f}L", delta=f"{net_cost / total_cost * 100:.1f}%")
        else:
            st.metric("✅ Surplus", f"₹{abs(net_cost):.1f}L", delta="No loan needed")

# ================ PROGRESS TRACKER ================
elif step == "📊 Progress Tracker":
    st.header("📊 Track Your Preparation Journey")

    # Exam preparation tracker
    st.subheader("📚 Exam Preparation Status")

    exams = ['GRE Verbal', 'GRE Quant', 'TOEFL', 'GATE', 'IELTS']
    progress = []

    col1, col2 = st.columns(2)
    with col1:
        for i, exam in enumerate(exams):
            if i < 3:
                progress.append(st.slider(f"{exam} Preparation", 0, 100, 60, key=exam))
    with col2:
        for i, exam in enumerate(exams):
            if i >= 3:
                progress.append(st.slider(f"{exam} Preparation", 0, 100, 40, key=exam))

    # Progress visualization
    fig = go.Figure(go.Bar(
        x=exams,
        y=progress,
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    ))
    fig.update_layout(title="Preparation Progress Overview", yaxis_title="Progress (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Application checklist
    st.subheader("✅ Application Checklist")

    checklist_items = [
        'University Research Completed',
        'SOP Draft Ready',
        'LOR Requests Sent',
        'Transcripts Obtained',
        'Financial Documents Prepared',
        'Visa Documentation Started'
    ]

    completed_items = []
    for item in checklist_items:
        if st.checkbox(item, key=f"check_{item}"):
            completed_items.append(item)

    completion_rate = 0
    if checklist_items:
        completion_rate = len(completed_items) / len(checklist_items) * 100

    st.progress(completion_rate / 100)
    st.write(f"Overall Progress: {completion_rate:.1f}%")

    if completion_rate == 100:
        st.balloons()
        st.success("🎉 Congratulations! You're ready to apply!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🎓 GradGuide - Empowering your academic journey with AI</p>
    <p>Made with ❤ for aspiring graduate students</p>
</div>
""", unsafe_allow_html=True)
