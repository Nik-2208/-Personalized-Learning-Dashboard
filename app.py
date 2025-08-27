import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import time

st.set_page_config(page_title="🎓 Personalized Learning Dashboard", layout="wide")
st.title("🎓 Personalized Learning Path Generator")
st.markdown(
    "Adjust your habits and learning style to see **predicted grades** and **personalized recommendations** in real-time!"
)

# Load models
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# --- INPUT SECTION ---
st.header("📝 Customize Your Learning Profile")

with st.expander("👤 Personal Info", expanded=True):
    col1, col2 = st.columns(2)
    with col1: age = st.slider("Age 🎂", 10, 100, 20)
    with col2: gender_en = st.selectbox("Gender 🧑", [0,1,2], format_func=lambda x: ["Male","Female","Other"][x])

with st.expander("📚 Study Habits", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1: study_hours = st.slider("Study Hours/Week", 0, 50, 15)
    with col2: attendance = st.slider("Attendance Rate (%)", 0, 100, 90)
    with col3: participation = st.select_slider("Participate in Discussions?", ["No","Yes"], value="Yes")

with st.expander("💆 Lifestyle & Wellness", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1: stress = st.slider("Stress Level 😫", 0, 10, 5)
    with col2: sleep = st.slider("Sleep Hours/Night 💤", 0, 12, 7)
    with col3: social_media = st.slider("Social Media Hours/Week 📱", 0, 50, 5)

with st.expander("🎨 Learning Style Preferences", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1: learning_style_kinesthetic = st.select_slider("Kinesthetic 👐", ["No","Yes"], value="No")
    with col2: learning_style_rw = st.select_slider("Reading/Writing 📖", ["No","Yes"], value="Yes")
    with col3: learning_style_visual = st.select_slider("Visual 🎨", ["No","Yes"], value="Yes")

# --- PREPARE DATA ---
input_df = pd.DataFrame({
    "Age": [age],
    "Study_Hours_per_Week": [study_hours],
    "Participation_in_Discussions_en": [1 if participation=="Yes" else 0],
    "Attendance_Rate (%)": [attendance],
    "Self_Reported_Stress_Level_en": [stress],
    "Time_Spent_on_Social_Media (hours/week)": [social_media],
    "Sleep_Hours_per_Night": [sleep],
    "Gender_en": [gender_en],
    "Preferred_Learning_Style_Kinesthetic": [1 if learning_style_kinesthetic=="Yes" else 0],
    "Preferred_Learning_Style_Reading/Writing": [1 if learning_style_rw=="Yes" else 0],
    "Preferred_Learning_Style_Visual": [1 if learning_style_visual=="Yes" else 0]
})

predicted_grade = rf_model.predict(input_df)[0]
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(input_df)

def generate_recommendations(df):
    advice = []
    if df["Study_Hours_per_Week"][0] < 20: 
        advice.append("📚 Increase weekly study hours for better retention.")
    if df["Participation_in_Discussions_en"][0] == 0: 
        advice.append("💬 Participate in discussions or forums.")
    if df["Attendance_Rate (%)"][0] < 90: 
        advice.append("🏫 Attend classes more consistently.")
    if df["Self_Reported_Stress_Level_en"][0] > 5: 
        advice.append("🧘 Reduce stress via mindfulness or breaks.")
    if df["Time_Spent_on_Social_Media (hours/week)"][0] > 10: 
        advice.append("📵 Limit social media while studying.")
    if df["Preferred_Learning_Style_Kinesthetic"][0] == 1: 
        advice.append("👐 Include hands-on activities.")
    if df["Preferred_Learning_Style_Visual"][0] == 1: 
        advice.append("🎨 Use diagrams, videos, and visual aids.")
    if df["Preferred_Learning_Style_Reading/Writing"][0] == 1: 
        advice.append("📝 Read and summarize notes.")

    # Determine dominant learning style
    dominant_style = max({
        'Kinesthetic': df["Preferred_Learning_Style_Kinesthetic"][0],
        'Visual': df["Preferred_Learning_Style_Visual"][0],
        'Reading/Writing': df["Preferred_Learning_Style_Reading/Writing"][0]
    }, key=lambda k: {
        'Kinesthetic': df["Preferred_Learning_Style_Kinesthetic"][0],
        'Visual': df["Preferred_Learning_Style_Visual"][0],
        'Reading/Writing': df["Preferred_Learning_Style_Reading/Writing"][0]
    }[k])

    return dominant_style, advice


# --- TABS ---
tabs = st.tabs(["Prediction 🏆", "Recommendations 📌", "Feature Importance 🔍"])

# --- PREDICTION TAB ---
with tabs[0]:
    st.subheader("🎯 Live Gamified Grade Dashboard")
    grade_container = st.empty()
    gauge_container = st.empty()
    progress_container = st.empty()
    kpi_container = st.empty()
    feedback_container = st.empty()

    def update_dashboard():
        predicted_grade_val = rf_model.predict(input_df)[0]
        if predicted_grade_val >= 90: grade_color="#1a8f00"; emoji="🌟 Excellent!"
        elif predicted_grade_val >= 80: grade_color="#4caf50"; emoji="👍 Good"
        elif predicted_grade_val >= 70: grade_color="#ffb74d"; emoji="🙂 Average"
        elif predicted_grade_val >= 60: grade_color="#ff9800"; emoji="⚠️ Below Average"
        else: grade_color="#f44336"; emoji="❌ Needs Improvement"

        # Grade card
        grade_container.markdown(f"""
            <div style='display:flex; justify-content:center; margin-bottom:20px;'>
                <div style='background: linear-gradient(135deg,#90caf9,#42a5f5);
                            padding:35px; border-radius:25px; text-align:center;
                            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
                            transform: perspective(600px) rotateX(5deg) rotateY(5deg);'>
                    <h1 style='color:{grade_color}; font-size:90px; margin:0;
                               padding:15px; border-radius:20px; 
                               background: rgba(0,0,0,0.15); display:inline-block;
                               box-shadow: 0 6px 15px rgba(0,0,0,0.25);
                               transition: transform 0.5s;'>{predicted_grade_val:.2f}</h1>
                    <h3 style='margin:10px; font-size:30px; 
                               background: rgba(0,0,0,0.08); padding:8px 20px; 
                               border-radius:15px; display:inline-block; color:#000;'>{emoji}</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_grade_val,
            delta={'reference':75,'increasing':{'color':'green'},'decreasing':{'color':'red'}},
            gauge={'axis':{'range':[0,100]},'bar':{'color':grade_color},
                   'steps':[{'range':[0,60],'color':'#e57373'},{'range':[60,70],'color':'#ffb74d'},
                            {'range':[70,80],'color':'#ffd54f'},{'range':[80,90],'color':'#aed581'},
                            {'range':[90,100],'color':'#388e3c'}],
                   'threshold':{'line':{'color':'blue','width':6},'thickness':0.8,'value':predicted_grade_val}})
        )
        fig.update_layout(height=400, margin={'t':0,'b':0,'l':0,'r':0})
        gauge_container.plotly_chart(fig, use_container_width=True)

        # Progress ring
        next_grade = 90 if predicted_grade_val<90 else 100
        progress_percent = int((predicted_grade_val/next_grade)*100)
        fig2 = go.Figure(go.Pie(values=[progress_percent,100-progress_percent], hole=0.7,
                                marker_colors=[grade_color,'#d3d3d3'], textinfo='none'))
        fig2.update_layout(height=250, width=250, showlegend=False,
                           annotations=[dict(text=f"{progress_percent}%", x=0.5,y=0.5,font_size=25,showarrow=False,font_color="#000")])
        progress_container.markdown("<div style='text-align:center; margin-top:20px;'><b>Progress to Next Grade</b></div>", unsafe_allow_html=True)
        progress_container.plotly_chart(fig2, use_container_width=True)

        # KPI Cards
        metrics = {"Study Hours/Week": study_hours,"Attendance Rate (%)":attendance,
                   "Participation":participation,"Stress Level":stress}
        kpi_cols = kpi_container.columns(len(metrics))
        for col, (label, value) in zip(kpi_cols, metrics.items()):
            col.markdown(f"""
                <div style='background: linear-gradient(135deg,#ffffffdd,#90caf9dd);
                            border-radius:25px; padding:25px; text-align:center;
                            box-shadow:0 8px 20px rgba(0,0,0,0.3);
                            transform: perspective(500px) rotateX(2deg) rotateY(2deg);
                            transition: all 0.5s;'>
                    <h4 style='margin:0; font-size:18px; color:#0d47a1;'>{label}</h4>
                    <p style='font-size:28px; color:#1a237e; margin:10px 0;'>{value}</p>
                </div>
            """, unsafe_allow_html=True)

        # Feedback
        if predicted_grade_val<70:
            feedback_container.markdown("<div style='text-align:center; font-size:18px'>💪 Keep pushing! Improve study habits to raise your grade.</div>", unsafe_allow_html=True)
        elif predicted_grade_val<90:
            feedback_container.markdown("<div style='text-align:center; font-size:18px'>🚀 Great! Stay consistent to reach the next level.</div>", unsafe_allow_html=True)
        else:
            feedback_container.markdown("<div style='text-align:center; font-size:18px'>🏆 Excellent work! You're a top performer! 🎉🌟</div>", unsafe_allow_html=True)

    update_dashboard()



# --- RECOMMENDATIONS TAB ---
with tabs[1]:
    st.subheader("📌 Personalized Recommendations")

    dominant_style, recommendations_list = generate_recommendations(input_df)
    st.markdown(f"""
    <div style='text-align:center; font-size:20px; margin-bottom:15px; color:#f5f5f5; 
                font-weight:bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.6);'>
        🎨 <b>Dominant Learning Style:</b> {dominant_style}
    </div>
""", unsafe_allow_html=True)

    if recommendations_list:
        categories = {
            "Study Habits": ["Study_Hours_per_Week", "Participation_in_Discussions_en", "Attendance_Rate (%)"],
            "Stress & Lifestyle": ["Self_Reported_Stress_Level_en", "Time_Spent_on_Social_Media (hours/week)", "Sleep_Hours_per_Night"],
            "Learning Style Tips": ["Preferred_Learning_Style_Kinesthetic", "Preferred_Learning_Style_Visual", "Preferred_Learning_Style_Reading/Writing"]
        }

        category_icons = {
            "Study Habits": "📚",
            "Stress & Lifestyle": "💆",
            "Learning Style Tips": "🎨"
        }

        rec_category_map = {}
        for rec in recommendations_list:
            if "study" in rec.lower() or "participate" in rec.lower() or "attend" in rec.lower():
                rec_category_map.setdefault("Study Habits", []).append(rec)
            elif "stress" in rec.lower() or "social media" in rec.lower() or "sleep" in rec.lower():
                rec_category_map.setdefault("Stress & Lifestyle", []).append(rec)
            else:
                rec_category_map.setdefault("Learning Style Tips", []).append(rec)

        for cat, recs in rec_category_map.items():
            container = st.container()
            container.markdown(f"""
                <div style='background: linear-gradient(135deg,#fff9c4,#ffe57f);
                            border-radius:25px; padding:20px; margin-bottom:20px;
                            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                            transform: perspective(600px) rotateX(3deg) rotateY(3deg);'>
                    <h3 style='margin-bottom:15px; color:#111;'>{category_icons.get(cat,'')} <b>{cat}</b></h3>
                </div>
            """, unsafe_allow_html=True)

            rec_cols = container.columns(len(recs))
            for c, rec in zip(rec_cols, recs):
                c.markdown(f"""
                    <div style='background: linear-gradient(135deg,#90caf9,#42a5f5);
                                border-radius:20px; padding:20px; text-align:center;
                                box-shadow: 0 6px 20px rgba(0,0,0,0.25);
                                transition: transform 0.3s ease-in-out;
                                transform: perspective(500px) rotateX(2deg) rotateY(2deg);
                                margin:5px;'>
                        <p style='margin:0; font-size:16px; color:#000; font-weight:bold;'>
                        {rec}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='background: linear-gradient(135deg,#c8e6c9,#81c784);
                        border-radius:25px; padding:30px; text-align:center;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                        transform: perspective(600px) rotateX(3deg) rotateY(3deg);'>
                <p style='color:#000; font-size:18px; font-weight:bold;'>✅ Your current habits are strong! Keep it up! 🎉</p>
            </div>
        """, unsafe_allow_html=True)



# --- FEATURE IMPORTANCE TAB ---
with tabs[2]:
    st.subheader("🔍 Feature Importance (SHAP Values)")

    shap_values_abs = abs(shap_values[0])
    feature_importance = pd.DataFrame({
        "Feature": input_df.columns,
        "Impact": shap_values_abs
    }).sort_values(by="Impact", ascending=True)

    colors = ['#1f77b4' if v>0 else '#ff7f0e' for v in shap_values[0]]

    fig = go.Figure(go.Bar(
        y=feature_importance["Feature"],
        x=feature_importance["Impact"],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='#333', width=1.5)
        ),
        hovertext=[f"{f}: {i:.2f}" for f, i in zip(feature_importance["Feature"], feature_importance["Impact"])],
        hoverinfo="text"
    ))

    fig.update_layout(
        height=500,
        margin=dict(l=30,r=30,t=30,b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_text="Feature Importance (SHAP Values)",
        title_font_size=22,
        xaxis_title="Impact on Predicted Grade",
        yaxis=dict(tickfont=dict(size=14, color="#111")),
        xaxis=dict(tickfont=dict(size=14, color="#111"))
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        <div style='padding:20px; border-radius:25px; background: linear-gradient(135deg,#e0f7fa,#b2ebf2);
                    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                    transform: perspective(600px) rotateX(3deg) rotateY(3deg);
                    font-weight:bold; color:#111; font-size:16px;'>
        <b>How to read this:</b><br>
        🔹 Blue bars indicate features that <b>increase</b> your predicted grade.<br>
        🔸 Orange bars indicate features that <b>decrease</b> your predicted grade.<br>
        🟦 Longer bars = higher impact.<br>
        🌈 Hover over bars to see exact SHAP values.
        </div>
    """, unsafe_allow_html=True)
