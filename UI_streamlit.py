import streamlit as st
import tempfile
import os
import cv2

from image_segmentation_AGENT import road_agent

#page config
st.set_page_config(
    page_title="Agentic Road Segmentation System",
    layout="wide"
)

#css
st.markdown(
    """
    <style>

    /* ---------- Global ---------- */
    .stApp {
        background:
            linear-gradient(180deg, #0B0F14 0%, #05070A 100%);
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0C1117 0%,
            #0A0E13 100%
        );
        border-right: 1px solid rgba(118,185,0,0.25);
    }

    /* ---------- Headings ---------- */
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: 0.4px;
        color: #EAEAEA;
    }

    h2 {
        border-bottom: 1px solid rgba(118,185,0,0.2);
        padding-bottom: 6px;
    }

    /* ---------- Chat Messages ---------- */
    .stChatMessage {
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 12px;
        backdrop-filter: blur(6px);
    }

    .stChatMessage[data-testid="chat-message-user"] {
        background: rgba(118,185,0,0.08);
        border: 1px solid rgba(118,185,0,0.35);
    }

    .stChatMessage[data-testid="chat-message-assistant"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
    }

    /* ---------- Chat Input ---------- */
    textarea {
        border-radius: 14px !important;
        border: 1px solid rgba(118,185,0,0.35) !important;
        background-color: rgba(255,255,255,0.03) !important;
    }

    /* ---------- Buttons ---------- */
    button {
        border-radius: 12px !important;
        background: rgba(118,185,0,0.08) !important;
        border: 1px solid rgba(118,185,0,0.4) !important;
        color: #EAEAEA !important;
        font-weight: 500;
    }

    button:hover {
        background: rgba(118,185,0,0.18) !important;
    }

    /* ---------- Metrics ---------- */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(118,185,0,0.25);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 0 20px rgba(118,185,0,0.05);
    }

    /* ---------- Images ---------- */
    img {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 0 25px rgba(0,0,0,0.5);
    }

    /* ---------- Divider ---------- */
    hr {
        border-top: 1px solid rgba(118,185,0,0.25);
    }

    </style>
    """,
    unsafe_allow_html=True
)

#title block
st.markdown(
     """
    ## 🟢 Agentic Satellite Road Segmentation System  
    **Autonomous perception, segmentation, and terrain reasoning from aerial imagery**

    <span style="color:rgba(118,185,0,0.85); font-size:0.9em;">
    Powered by agentic reasoning with explicit state
    </span>
    """,
    unsafe_allow_html=True
)

#initialise session
if "road_state" not in st.session_state:
    st.session_state.road_state = {
        "user_query": "",
        "image_path": None,
        "mask_path": None,
        "metrics": None,
        "interpretation": None,
        "intent": None,
        "final_response": None,
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#sidebar
with st.sidebar:
    st.header("Image Memory")

    st.caption(
        "Upload once. The agent will remember it "
        "across the entire conversation."
    )

    uploaded_file = st.file_uploader(
        "Drag & drop a satellite / aerial image",
        type=["png", "jpg", "jpeg", "tiff"]
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        st.session_state.road_state["image_path"] = image_path
        st.success("Image loaded into agent memory.")

    st.markdown("---")

    if st.button("Reset conversation"):
        del st.session_state["road_state"]
        del st.session_state["chat_history"]
        st.rerun()
    
    if st.session_state.road_state.get("image_path"):
        st.success("Image embedded in agent state")
    else:
        st.warning("No image loaded")

#main layout
left, right = st.columns([1.2, 1])

#chat interface
with left:
    st.subheader("Chat")

    #display chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    #Chat input
    user_input = st.chat_input(
        "Hi!! Ask me to segment, analyze, or reason about the road..."
    )

    if user_input:
        #add user message
        st.session_state.chat_history.append(("user", user_input))

        #update agent state
        st.session_state.road_state["user_query"] = user_input

        with st.spinner("Agent reasoning..."):
            st.session_state.road_state = road_agent.invoke(
                st.session_state.road_state
            )
        
        #assistant response
        assistant_reply = st.session_state.road_state.get(
            "final_response", "Done."
        )
        st.session_state.chat_history.append(
            ("assistant", assistant_reply)
        )

        st.rerun()

#visual outputs
with right:
    st.subheader("Visual Analysis")

    if st.session_state.road_state.get("image_path"):
        img = cv2.imread(st.session_state.road_state["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption="Original Image", use_container_width=True)
    
    if st.session_state.road_state.get("mask_path"):
        mask = cv2.imread(
            st.session_state.road_state["mask_path"],
            cv2.IMREAD_GRAYSCALE
        )
        st.image(mask, caption="Road segmentation mask", use_container_width=True)
    
    if st.session_state.road_state.get("metrics"):
        st.markdown("### Quantitative Evaluation")

        m = st.session_state.road_state["metrics"]

        st.metric("Road coverage", f"{m['road_coverage']:.3f}")
        st.metric("Avg width (m)", f"{m['avg_width_m']:.2f}")
        st.metric("Junction density", f"{m['junction_density']:.3f}")
        st.metric("Fragmentation", f"{m['fragmentation']:.5f}")
        st.metric("Confidence", f"{m['confidence']:.2f}")

#footer
st.markdown("---")
st.caption(
    "This chatbot maintains explicit structured state across turns, "
    "allowing multi-step reasoning without relying on implicit LLM memory."
)