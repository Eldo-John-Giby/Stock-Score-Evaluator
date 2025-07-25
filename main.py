from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import streamlit as st
import plotly.graph_objects as go
import ast
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Function to get LLM response
def get_response(stock):
    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama3-70b-8192",
        temperature=0,
        stop=["\n\n"]
    )

    prompt_template = PromptTemplate(
        input_variables=["stock"],
        template="""
        Do a fundamental analysis of {stock} and assign a score out of 10 for each of the following parameters.
        Be truthful in analysis, no sugarcoating , only Truth

        Return the final scores **only** as a valid Python dictionary. Each parameter must be scored out of 10.

        Scoring Table:
        No. Parameter                     Weight   Description
        1.  Business Quality              15%      Core industry relevance, moat, sustainability of business model.
        2.  Profitability                15%      Net profit margin, ROE, ROA.
        3.  Growth Potential             20%      Revenue/profit CAGR, industry tailwinds, expansion plans.
        4.  Asset Quality / Leverage     10%      Debt levels, interest coverage, debt-to-equity.
        5.  Cost Efficiency              10%      Operating margin, cost ratios, overhead control.
        6.  Valuation                    10%      P/E, P/B, EV/EBITDA vs peers.
        7.  Management & Governance      10%      Promoter history, auditors, transparency, related party transactions.
        8.  Promoter Skin in the Game    5%       Shareholding levels, pledge status.
        9.  Liquidity & Public Float     5%       Trading volumes, institutional holding, ease of entry/exit.

        Instructions:
        - No explanations or extra commentary.
        - Respond with only the dictionary. No extra characters. 
        - Include an additional key at the end: "Overall score" 
        - Use this formula for **Overall score**:  
          **Overall score = Σ (parameter_score × weight)**  
          (Weights: 15%, 15%, 20%, 10%, 10%, 10%, 10%, 5%, 5%)
        """

    )

    chain = prompt_template | llm
    response = chain.invoke({"stock": stock})
    return response.content

# Extract dictionary safely from response
def extract_dict(response_str):
    try:
        match = re.search(r"\{.*?\}", response_str, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0))
    except Exception as e:
        st.error(f"⚠️ Parsing failed: {e}")
    return None

# Show a gauge for a specific score
def show_gauge(score, title=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "/10", 'font': {'size': 28}},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {
                'range': [0, 10],
                'tickwidth': 1,
                'tickcolor': "gray",
                'tickmode': 'linear',
                'tick0': 0,
                'dtick': 2
            },
            'bar': {'color': "dodgerblue", 'thickness': 0.3},
            'steps': [
                {'range': [0, 3], 'color': "#e74c3c"},
                {'range': [3, 6], 'color': "#f39c12"},
                {'range': [6, 8], 'color': "#f1c40f"},
                {'range': [8, 10], 'color': "#27ae60"},
            ],
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.update_layout(
        margin=dict(t=60, b=60, l=40, r=40),  # ⬅️ More margin to prevent clipping
        paper_bgcolor="#0e1117",
        font={'color': "white"},
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.set_page_config(page_title="📊 Fundamental Stock Analyzer", page_icon="📈", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 2.5rem; }
        .css-1d391kg, .css-1lcbmhc { padding-top: 1rem; }
        .main { background: linear-gradient(to right, #e0f7fa, #fff); font-family: 'Segoe UI', sans-serif; }
        .stTextInput>div>div>input {
            border-radius: 12px; padding: 10px; border: 1px solid #ccc;
        }
        .stButton>button {
            background-color: #00acc1; color: white; font-weight: bold;
            border-radius: 8px; padding: 10px 20px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover { background-color: #00838f; }
        .stMarkdown h1, .stMarkdown h2 { color: #006064; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("## 🔍 Fundamental Stock Analyzer")
st.markdown("Use the power of AI to get a quick, no-nonsense stock analysis.")

# Input
stock_input = st.text_input("💼 Enter stock/company name", value="", placeholder="e.g., Infosys")

# Trigger
if st.button("🚀 Analyze Now"):
    if stock_input.strip() == "":
        st.warning("Please enter a valid stock name.")
    else:
        with st.spinner("Crunching financials and decoding fundamentals... 💻📊"):
            result = get_response(stock_input)
            st.success("✅ Analysis complete!")

            st.markdown("### 📄 AI-Powered Analysis")
            st.code(result, language="python")

            parsed_result = extract_dict(result)
            if parsed_result:
                if "Overall score" in parsed_result:
                    overall_score = parsed_result["Overall score"]
                    st.markdown(
                        f"""
                        <div style="text-align: center; font-size: 42px; font-weight: bold; color: #2196f3; margin-top: 2rem;">
                            📊 Overall Score: {overall_score}/10
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.json(parsed_result)

                st.markdown("### 🔧 Performance Gauges")

                gauge_items = [
                    (k, v) for k, v in parsed_result.items()
                    if k != "Overall score"
                ]

                for i in range(0, len(gauge_items), 4):
                    cols = st.columns(4)
                    for j, (key, value) in enumerate(gauge_items[i:i + 4]):
                        try:
                            score = float(value)
                            with cols[j]:
                                show_gauge(score, title=key)
                        except Exception as e:
                            st.warning(f"Could not render gauge for '{key}': {e}")
            else:
                st.error("❌ Failed to parse response into dictionary.")
