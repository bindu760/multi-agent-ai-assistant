"""
app.py
------
Streamlit chat UI. Yesle:
  - Groq API key sidebar bata lin/env bata lin
  - CV upload garna dincha (sidebar) -> uploads/ folder ma save huncha
  - Chat interface dincha, jaha user le jasto pani prasna sodhna sakcha
    (college admission / IT career / general web / "optimize my cv")
  - Conversation history session_state ma rakhcha ra agent lai context
    ko rup ma pass garcha
"""

import os
import streamlit as st

from crew_agent import run_agent
from tools import LATEST_CV_POINTER, UPLOADS_DIR, OUTPUTS_DIR

st.set_page_config(page_title="EduCareer AI Assistant", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}
if "cv_uploaded" not in st.session_state:
    st.session_state.cv_uploaded = os.path.exists(LATEST_CV_POINTER)

# ---------------------------------------------------------------------------
# Sidebar: API key + CV upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Setup")

    default_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input("Groq API Key", value=default_key, type="password",
                             help="Get a free key from console.groq.com")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()
    st.subheader("📄 Upload your CV")
    uploaded_file = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        save_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with open(LATEST_CV_POINTER, "w", encoding="utf-8") as f:
            f.write(save_path)
        st.session_state.cv_uploaded = True
        st.success(f"Uploaded: {uploaded_file.name}")
        st.caption('Now just ask in chat: "optimize my CV" or "optimize my CV for a backend developer role"')

    optimized_path = os.path.join(OUTPUTS_DIR, "optimized_cv.docx")
    if os.path.exists(optimized_path):
        with open(optimized_path, "rb") as f:
            st.download_button(
                "⬇️ Download optimized CV (.docx)",
                data=f.read(),
                file_name="optimized_cv.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "This assistant can help with:\n"
        "- 🏫 College admission questions\n"
        "- 💼 IT career paths, salary & future outlook\n"
        "- 🌐 Live web search for current info\n"
        "- 📄 CV upload → optimize → download"
    )

# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------
st.title("🎓 EduCareer AI Assistant")
st.caption("Powered by CrewAI + Groq (Llama 3.3 70B) — college admission, IT career guidance & CV optimization in one chat.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask about college admission, IT careers, or say 'optimize my CV'...")

if user_query:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter your Groq API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = run_agent(
                        query=user_query,
                        chat_history=st.session_state.messages[:-1],
                        api_key=os.environ["GROQ_API_KEY"],
                        has_uploaded_cv=st.session_state.cv_uploaded,
                    )
                except Exception as e:
                    answer = f"⚠️ Something went wrong: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Refresh sidebar download button if a CV was just optimized
        if os.path.exists(os.path.join(OUTPUTS_DIR, "optimized_cv.docx")):
            st.rerun()