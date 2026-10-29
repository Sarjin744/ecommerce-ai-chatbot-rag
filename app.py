import os
import streamlit as st
import pandas as pd
import altair as alt
from engine import EcommerceChatbot

st.set_page_config(
    page_title="AI E-Commerce Chatbot | Lab Exp 1",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #6c757d;
        margin-bottom: 20px;
    }
    .intent-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.82rem;
        margin-bottom: 10px;
    }
    .intent-faq { background-color: #e3f2fd; color: #0d47a1; }
    .intent-rec { background-color: #e8f5e9; color: #1b5e20; }
    .intent-sent { background-color: #f3e5f5; color: #4a148c; }
    .intent-qa { background-color: #fff9c4; color: #f57f17; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chatbot" not in st.session_state:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    st.session_state.chatbot = EcommerceChatbot(data_dir=data_dir)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

bot: EcommerceChatbot = st.session_state.chatbot

def handle_query(query_text: str):
    """Processes a user query and appends to chat history."""
    if not query_text or not query_text.strip():
        return
    with st.spinner("Analyzing intent and generating response..."):
        res = bot.process_query(query_text.strip())
        st.session_state.chat_history.append({"user": query_text.strip(), "bot": res})

# Sidebar: System Status & Datasets
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/shopping-cart.png", width=110)
    st.markdown("### 🏛️ Karunya University")
    st.markdown("**24CS2506 – Building Applications using GPT-4**")
    st.markdown("**Experiment 1: AI-Powered E-Commerce Chatbot**")
    st.markdown("---")
    
    st.subheader("⚙️ System Status")
    if bot.ollama.is_connected:
        st.success(f"● Ollama Active ({bot.ollama.model})")
    else:
        st.warning("○ Ollama Offline\n(Using Local Semantic Fallback & Scikit-learn)")
        
    st.markdown("---")
    if st.button("🗑️ Clear Conversation History"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.subheader("📁 Datasets Inspection")
    view_data = st.selectbox("Select Dataset to View", ["Products", "FAQs", "Reviews"])
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if view_data == "Products":
        df = pd.read_csv(os.path.join(data_dir, "products.csv"))
        st.dataframe(df[["Product Name", "Category", "Price"]], use_container_width=True)
    elif view_data == "FAQs":
        df = pd.read_csv(os.path.join(data_dir, "faqs.csv"))
        st.dataframe(df[["Question"]], use_container_width=True)
    elif view_data == "Reviews":
        df = pd.read_csv(os.path.join(data_dir, "reviews.csv"))
        st.dataframe(df[["Product", "Review"]], use_container_width=True)

# Main Title
st.markdown('<div class="main-header">🛍️ AI-Powered E-Commerce Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Intent Shopping Assistant with Semantic Search, TF-IDF Recommendations, Review Sentiment Analysis & Product Q&A</div>', unsafe_allow_html=True)

# Quick Sample Query Buttons - Instantly trigger query upon click!
st.markdown("##### ⚡ One-Click Test Queries:")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("💻 Recommend gaming laptop", use_container_width=True):
        handle_query("Recommend a gaming laptop under 145000")
        st.rerun()
with col2:
    if st.button("🌊 Is this phone waterproof?", use_container_width=True):
        handle_query("Is this phone waterproof?")
        st.rerun()
with col3:
    if st.button("📦 Where is my parcel?", use_container_width=True):
        handle_query("Where is my parcel?")
        st.rerun()
with col4:
    if st.button("🔄 How can I return order?", use_container_width=True):
        handle_query("How can I return my order?")
        st.rerun()
with col5:
    if st.button("⭐ Laptop reviews sentiment", use_container_width=True):
        handle_query("What do customers think about Laptop?")
        st.rerun()

st.markdown("---")

# Render conversation history
if not st.session_state.chat_history:
    st.info("💡 **Welcome!** Click any of the sample queries above or type your question below (e.g. *'Recommend an office laptop under 60000'*, *'Where is my parcel?'*, or *'How is the battery of iPhone 15 Pro?'*).")
else:
    for chat in st.session_state.chat_history:
        q = chat["user"]
        res = chat["bot"]
        
        # User message
        with st.chat_message("user", avatar="👤"):
            st.write(q)
            
        # Assistant message
        with st.chat_message("assistant", avatar="🛍️"):
            intent = res["intent"]
            conf = res["confidence"] * 100
            source = res["intent_source"]
            
            intent_class = {
                "faq": "intent-faq",
                "product_recommendation": "intent-rec",
                "sentiment_analysis": "intent-sent",
                "product_qa": "intent-qa"
            }.get(intent, "intent-faq")
            
            st.markdown(
                f'<span class="intent-pill {intent_class}">🎯 Intent: {intent.upper()} ({conf:.1f}%)</span> &nbsp; '
                f'<span style="color:#777; font-size:0.83rem;">Source: {source}</span>',
                unsafe_allow_html=True
            )
            
            res_type = res["type"]
            data = res["data"]
            
            if res_type == "faq":
                st.info(f"**FAQ Answer:** {res['text']}")
                with st.expander("🔍 Semantic Search Match Details"):
                    st.write(f"- **Matched FAQ Question:** {data.get('matched_question')}")
                    st.write(f"- **Cosine Similarity Score:** `{data.get('similarity_score')}`")
                    st.write(f"- **Search Method:** {data.get('method')}")
                    
            elif res_type == "recommendation":
                st.markdown(f"**{res['text']}**")
                recs = data.get("recommendations", [])
                if recs:
                    cols = st.columns(len(recs))
                    for idx, r in enumerate(recs):
                        with cols[idx]:
                            st.markdown(f"**{r['name']}**")
                            st.caption(f"Brand: {r['brand']} | Category: {r['category']}")
                            st.markdown(f"**Price:** `{r['price']}`")
                            st.markdown(f"**Match Score:** `{r['similarity_score']}`")
                            st.caption(r["description"])
                with st.expander("🔍 TF-IDF Extraction Details"):
                    st.write(f"- **Extracted Keywords:** `{', '.join(data.get('extracted_tfidf_keywords', []))}`")
                    st.write(f"- **Budget Constraint:** `{data.get('budget_filter')}`")
                    
            elif res_type == "sentiment":
                st.markdown(res['text'])
                counts = data.get("counts", {})
                
                chart_data = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Count": [counts.get("Positive", 0), counts.get("Negative", 0), counts.get("Neutral", 0)]
                })
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("Sentiment:N", sort=None),
                    y="Count:Q",
                    color=alt.Color("Sentiment:N", scale=alt.Scale(
                        domain=["Positive", "Negative", "Neutral"],
                        range=["#2e7d32", "#c62828", "#fbc02d"]
                    ))
                ).properties(height=180)
                st.altair_chart(chart, use_container_width=True)
                
                with st.expander("📝 Individual Reviews Breakdown"):
                    for rev in data.get("reviews", []):
                        badge = "🟢" if rev["sentiment"] == "Positive" else ("🔴" if rev["sentiment"] == "Negative" else "🟡")
                        st.markdown(f"{badge} **[{rev['sentiment']}]** {rev['review']}")
                        
            elif res_type == "product_qa":
                st.warning(f"🤖 **Product QA Assistant:** {res['text']}")
                with st.expander("🔍 Product Context & Specs"):
                    st.write(f"- **Identified Product:** {data.get('matched_product')}")
                    st.write(f"- **Brand:** {data.get('brand')}")
                    st.write(f"- **Price:** {data.get('price')}")
                    st.write(f"- **Engine:** {data.get('engine')}")

# Two Input Options: 
# Option A: Modern native st.chat_input at the bottom
chat_input_val = st.chat_input("Ask a question, request a product recommendation, or check reviews...")
if chat_input_val:
    handle_query(chat_input_val)
    st.rerun()

# Option B: Manual text box + Send button (in case user prefers form)
with st.container():
    c_input, c_btn = st.columns([5, 1])
    with c_input:
        manual_val = st.text_input("Or type here:", key="manual_text_input", label_visibility="collapsed", placeholder="Type your shopping query...")
    with c_btn:
        manual_send = st.button("Send Query 🚀", use_container_width=True, key="manual_send_btn")
    
    if manual_send and manual_val.strip():
        handle_query(manual_val)
        st.rerun()
