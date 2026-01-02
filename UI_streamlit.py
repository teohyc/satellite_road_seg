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

st.title("Agentic Road Segmentation Bot")
st.caption(
    "A conversational agent for road segmentation, analysis, "
    "and decision reasoning using explicit system state."
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
    st.header("Input Image")

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
        st.experimental_rerun()

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
        st.session_state.chat_history.append("user", user_input)

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

        st.experimental_rerun()

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
        st.markdown("### Metrics")

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