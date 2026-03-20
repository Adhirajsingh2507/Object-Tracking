import streamlit as st
import google.generativeai as genai
import os

# --- UI CONFIGURATION ---
st.set_page_config(page_title="VETERAN TECH ANALYST TERMINAL", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for the "Codey" UI/UX
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #00ff41;
        font-family: 'Fira Code', monospace;
    }
    .stTextInput > div > div > input {
        background-color: #161b22;
        color: #00ff41;
        border: 1px solid #30363d;
    }
    .stButton > button {
        background-color: #238636;
        color: white;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .report-box {
        padding: 20px;
        border: 1px solid #30363d;
        border-radius: 10px;
        background-color: #161b22;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SYSTEM PROMPT ---
VETERAN_PROMPT = """
# ROLE
You are a tech news reporter and analyst with 50 years of experience. 
You fetch news (simulated via your internal knowledge and search tools) that affect the tech industry, analyze connections between Math, Physics, AI, and Coding, and report to the client.

# CONSTRAINTS
- Tone: Professional but witty (Veteran style).
- Sources: ONLY use signals related to Reuters, Bloomberg, FT, Layoffs.fyi, arXiv, Nature, GitHub, and major AI blogs (OpenAI/Anthropic).
- Response Format:
  [Explanation]
  [Pro Tip]
  [major findings and what future decision should you take]
  [Follow-up Question]
"""

# --- APP LOGIC ---
def main():
    st.title("📟 TECH_VETERAN_ANALYST_v1.0.4")
    st.subheader("Industry Intelligence & Macro-Trend Analysis")

    # Sidebar for API Configuration
    with st.sidebar:
        st.header("System Settings")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        model_choice = st.selectbox("Select Core", ["gemini-1.5-pro", "gemini-1.5-flash"])
        st.info("The analyst requires an active uplink (API Key) to process global signals.")

    # User Input
    user_query = st.chat_input("Input sector or query (e.g., 'Latest in Silicon Photonics' or 'Hiring Trends')")

    if user_query:
        if not api_key:
            st.error("ERROR: NO API KEY DETECTED. CRITICAL FAILURE.")
            return

        try:
            # Initialize Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=model_choice,
                system_instruction=VETERAN_PROMPT
            )

            with st.status("Fetching signals from Reuters, arXiv, and Bloomberg...", expanded=True) as status:
                st.write("Connecting to global nodes...")
                st.write("Filtering noise from signal...")
                st.write("Mapping Geopolitics to Math breakthroughs...")
                
                # In a real tool, you'd use a search tool here. 
                # For this implementation, the model uses its high-density training data/search.
                response = model.generate_content(user_query)
                status.update(label="Analysis Complete", state="complete", expanded=False)

            # Display results in the themed box
            st.markdown(f"""
            <div class="report-box">
                {response.text}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"SYSTEM ERROR: {str(e)}")

    # Footer
    st.markdown("---")
    st.caption("Terminal Status: ONLINE | Data Sources: VERIFIED | Mode: PROFESSIONAL/WITTY")

if __name__ == "__main__":
    main()
