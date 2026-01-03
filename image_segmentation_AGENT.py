from typing import TypedDict, Optional, Literal
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from road_seg_infer import ConvBlock, UNet, road_seg_infer_main, load_image, predict_mask, visualize_result
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import convolve, label
import cv2
import os

#for tree display purpoe only
'''
from PIL import Image
from io import BytesIO'''



ROUTER_MODEL_NAME = "qwen3:0.6b"
EXPLAINER_MODEL_NAME = "deepseek-r1:1.5b"

#define local llm used
router_llm = ChatOllama(
    model=ROUTER_MODEL_NAME,
    temperature=0.0
)

explainer_llm = ChatOllama(
    model=EXPLAINER_MODEL_NAME,
    temperature=0.2
)

#agent state
class RoadState(TypedDict):
    user_query: str
    image_path: Optional[str]

    mask_path: Optional[str]
    metrics: Optional[dict]
    interpretation: Optional[dict]

    intent: Optional[str]
    final_response: Optional[str]

#segmentation tool
def run_road_segmentation(image_path: str) -> str:
    return road_seg_infer_main(image_path)

#analyzer tool
def analyze_road_mask(mask_path: str) -> dict:
    
    #load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f'Failed to load mask from {mask_path}')

    #binarize mask
    _, mask_bin = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
    mask_bin = mask_bin.astype(bool)

    #road coverage
    area_px = mask_bin.sum()
    total_px = mask_bin.size
    road_coverage = area_px / total_px

    #skeletonize
    skeleton = skeletonize(mask_bin)
    length_px = skeleton.sum() 
    length_m = length_px * 0.5

    #average road width
    area_m2 = area_px * (0.5 ** 2) #assume meter per pixel is 0.5
    avg_width_m = area_m2 / (length_m + 1e-6)

    #junction detection by convolution
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])
    neighbors = convolve(skeleton.astype(int), kernel, mode="constant")
    junctions = skeleton & (neighbors >= 3)
    num_junctions = int(junctions.sum())
    junction_density = num_junctions / (length_m + 1e-6)

    #connected components
    _, num_components = label(mask_bin) #find disconnected fragments

    #fragmentation index
    fragmentation = num_components / (area_px + 1e-6)
    
    #confidence
    confidence = 1.0
    if road_coverage < 0.01:
        confidence *= 0.5
    if num_components > 20:
        confidence *= 0.6
    if length_px < 50:
        confidence *= 0.4
    return {
        "road_coverage": float(road_coverage),
        "avg_width_m": float(avg_width_m),
        "junction_density": float(junction_density),
        "num_components": int(num_components),
        "fragmentation": float(fragmentation),
        "confidence": float(confidence)
    }

#interpreter tool
def interpret_metrics(metrics: dict) -> dict:
    if metrics["road_coverage"] > 0.15:
        area = "Urban"
    elif metrics["road_coverage"] > 0.05:
        area = "Suburban"
    else:
        area = "Rural"

    return {
        "area_type": area,
        "confidence": metrics["confidence"]
    }

#router llm helper (START)
def intent_router(state: RoadState) -> RoadState:
    prompt = f"""
You are an intent classifier.

User query:"{state['user_query']}"

System state:
- image path: {state.get('image_path') is not None}
- mask path: {state.get('mask_path') is not None}
- metrics: {state.get('metrics') is not None}

Rules:
- If mask path is None, you MUST choose "segment" regardless of whatever content is in user query.
- If metrics exist, you MAY choose "explain"
- If mask exists but metrics do not, choose "analyze"
- If the user query is a question statement without the word "generate", "segment", the option "segment" shall NEVER be chosen
- If user tells you to segment /analyse /renanalyse/ resegment/ process an image or a new image, choose "segment"
- If user suggest an existence of a new image, choose "segment"
 
Choose ONE out of the THREE:
 segment (Note: must be chosen no matter what the user"s query is when the System state shows that the mask path is None or empty or does not have a .png path)
 analyze
 explain

Respond with ONE WORD only.
"""

    intent = router_llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    
    #safety override
    '''if intent == "analyze" and not state.get("mask_path"):
        intent = "segment"'''

    
    print(f"Intent: {intent}")
    state["intent"] = intent
    return state

#defining nodes
def segment_node(state: RoadState) -> RoadState:
    state["mask_path"] = run_road_segmentation(state["image_path"])
    return state


def analyze_node(state: RoadState) -> RoadState:
    if not state.get("mask_path"):
        raise ValueError(
            "Analyze node called without existing mask_path"
        )

    state["metrics"] = analyze_road_mask(state["mask_path"])
    return state


def interpret_node(state: RoadState) -> RoadState:
    state["interpretation"] = interpret_metrics(state["metrics"])
    return state

def explain_node(state: RoadState) -> RoadState:
    prompt = f"""
You are a road-network analysis assistant.
Pretend you are able to analyse an aerial satellite image based on the "Computed metrics" and "Derived interpretation".

User question:
"{state['user_query']}"

Computed metrics:
{state['metrics']}

Derived interpretation:
{state['interpretation']}

Rules:
- Prioritise in answering the user's query first
- Base conclusions on the metrics
- Do NOT invent data for the metrics
- Explain clearly 
- Mention uncertainty if confidence is low
- When answering user's query always used the computed metrics provided.
"""

    msg = HumanMessage(content=prompt)
    response = explainer_llm.invoke([msg]).content

    state["final_response"] = response
    return state

#router function
def route_from_intent(state: RoadState) -> Literal["segment", "analyze", "explain"]:
    return state["intent"]

#building graph
graph = StateGraph(RoadState)

graph.add_node("intent_router", intent_router)
graph.add_node("segment", segment_node)
graph.add_node("analyze", analyze_node)
graph.add_node("interpret", interpret_node)
graph.add_node("explain", explain_node)

graph.set_entry_point("intent_router")

graph.add_conditional_edges(
    "intent_router",
    route_from_intent,
    {
        "segment": "segment",
        "analyze": "analyze",
        "explain": "explain",
    },
)

graph.add_edge("segment", "analyze")
graph.add_edge("analyze", "interpret")
graph.add_edge("interpret", "explain")
graph.add_edge("explain", END)

road_agent = graph.compile()

#initializing
state = {
    "user_query": "",
    "image_path": "infer_satellite_image.tiff",

    "mask_path": None,
    "metrics": None,
    "interpretation": None,

    "intent": None,
    "final_response": None,
}

#the tree diagram
'''png = road_agent.get_graph().draw_mermaid_png()
img = Image.open(BytesIO(png))
img.show()
img.save("road_seg_agent_diagram.png")'''

'''
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    state["user_query"] = user_input
    state = road_agent.invoke(state)

    print("\nAssistant:")
    print(state["final_response"])
'''

__all__ = [
    "road_agent",
    "ROUTER_MODEL_NAME",
    "EXPLAINER_MODEL_NAME"
]
    