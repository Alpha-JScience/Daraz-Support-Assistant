import streamlit as st

DARAZ_ORANGE = "#F85606"
DARAZ_DARK = "#1A1A1A"


def apply_custom_css():
    """Applies Daraz-branded custom CSS to the Streamlit app."""
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: #FAFAFA; }}
        header[data-testid="stHeader"] {{ background-color: {DARAZ_ORANGE}; }}
        .daraz-banner {{
            background: linear-gradient(90deg, {DARAZ_ORANGE} 0%, #FF8A3D 100%);
            padding: 18px 24px;
            border-radius: 12px;
            margin-bottom: 18px;
        }}
        .daraz-banner h1 {{ color: white; font-size: 26px; margin: 0; font-weight: 700; }}
        .daraz-banner p {{ color: #FFE7D6; margin: 4px 0 0 0; font-size: 14px; }}
        section[data-testid="stSidebar"] {{ background-color: {DARAZ_DARK}; }}
        section[data-testid="stSidebar"] * {{ color: #F2F2F2 !important; }}
        .stChatMessage {{ border-radius: 10px; }}
        div[data-testid="stChatInput"] textarea {{ border: 1px solid {DARAZ_ORANGE} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_banner():
    """Renders the main Daraz banner."""
    st.markdown(
        """
        <div class="daraz-banner">
            <h1>🛍️ Daraz Support Assistant</h1>
            <p>Ask about returns, delivery, refunds, seller policies, payments & customer support</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(sections: dict) -> tuple[str, str | None]:
    """Renders the sidebar and returns the selected label and department value."""
    with st.sidebar:
        st.markdown("### 📚 Knowledge Base Section")
        st.caption("Restrict search to one section, or search everything.")
        selected_label = st.radio(
            "Section",
            options=list(sections.keys()),
            index=0,
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    return selected_label, sections[selected_label]


def render_chat_history():
    """Renders the existing chat history from session state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"- **{s['department']}** — `{s['source_file']}`")
