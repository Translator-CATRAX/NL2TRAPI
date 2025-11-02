# #!/usr/bin/env python3
# # Streamlit demo for NL2TRAPI-langgraph (enhanced)

# import json
# import logging
# import time
# from io import StringIO
# from typing import Any, Dict, List, Tuple

# import streamlit as st

# # Graph viz (optional)
# try:
#     from pyvis.network import Network  # pip install pyvis
#     _HAS_PYVIS = True
# except Exception:
#     _HAS_PYVIS = False

# from trapi_agent.agent_graph import graph

# # ─────────────────────────── UI OPTIONS ───────────────────────────
# ROUTES = ["Auto (router picks)", "onehop", "pathfinder", "treats", "chem_gene", "xcrg","pathfinder_constrained"]

# EXAMPLES: List[Tuple[str, str]] = [
#     ("onehop", "What proteins does acetaminophen interact with?"),
#     ("onehop", "What biological processes are related to GFAP?"),
#     ("pathfinder", "Find a path between asthma and diabetes mellitus"),
#     ("pathfinder", "By what paths are ibuprofen and headaches connected?"),
#     ("treats", "What drugs treat asthma?"),
#     ("treats", "What drugs treat diabetes mellitus?"),
#     ("treats", "What drugs treat castleman disease?"),
#     ("treats", "What drugs treat What drugs treat malignant ciliary body melanoma?"),
#     ("treats", "What chemicals are predicted to be useful to treat malignant ciliary body melanoma?"),
#     ("pathfinder", "How are neutropenia and filgrastim related in multi-hop paths?"),
#     ("pathfinder","Find me paths between ibuprofen and COX1?"),
#     ("xcrg","what genes are upregulated by filgrastim?"),
#     ("xcrg","what genes are downregulated by filgrastim?"),
#     ("xcrg","which drugs inhibit the activity of ABCB1"),
#     ("pathfinder_constrained", "Find a path between asthma and diabetes mellitus that includes a protein"),
#     ("pathfinder_constrained", "Find a path between asthma and diabetes mellitus that includes a biological_process"),
#     ('pathfinder_constrained', "How are ibuprofen and headaches related via paths going through genes"),
#     ('pathfinder_constrained', "By what paths are BRCA1 and breast cancer connected via genes"),
#     ('pathfinder_constrained', "Find me paths between ibuprofen and COX1 via proteins"),
#     ('pathfinder_constrained', "Find me paths between ibuprofen and COX1 via biological_processes"),
#     ('pathfinder_constrained', "Find me paths between ibuprofen and COX1 via diseases"),
#     ('pathfinder_constrained', "How are EGFR and lung cancer related through proteins?"),   
#     ('pathfinder_constrained', "How are neutropenia and filgrastim related in multi-hop paths going through diseases?"),
#     ('pathfinder_constrained', "How are obesity and insulin resistance connected via diseases?"),
#     ('pathfinder_constrained', "Show connections between BRCA1 and asthma via a drug?"),    
#     ('pathfinder_constrained', "Paths between TNF and rheumatoid arthritis via a chemical?"),
#     ('pathfinder_constrained', "Find paths from LRRK2 to Parkinson disease via small molecules?"),  
#     ('pathfinder_constrained', "Show multi-hop paths between BRCA1 and DNA repair through pathways?"),  
#     ('pathfinder_constrained', "Show paths from kinase inhibitors to EGFR via molecular activities?"),  
#     ('pathfinder_constrained', "How are COX1 and prostaglandin synthesis related through activities?"),
#     ('pathfinder_constrained', "Find paths between GFAP and seizures via tissues?"),
#     ('pathfinder_constrained', "Show paths between HIF1A and hypoxia via cells?"),
#     ('pathfinder_constrained', "How are BRCA1 and DNA repair related through organelles?"),
#     ('pathfinder_constrained', "Show paths between APOE and Alzheimer disease through phenotypes)"),
#     ('pathfinder_constrained', "Find paths from TP53 to cancer via phenotypes)"),
    

# ]

# logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)

# st.set_page_config(page_title="NL → TRAPI Demo", layout="wide")
# st.title("🧠 NL → TRAPI Demo")

# # minimal session state
# if "history" not in st.session_state:
#     st.session_state.history = []  # list of dicts

# # ─────────────────────────── Sidebar ───────────────────────────
# with st.sidebar:
#     st.header("Settings")
#     route_choice = st.selectbox("Route", ROUTES, index=0)
#     ex_label = st.selectbox(
#         "Examples",
#         ["(pick one)"] + [f"{r} · {q}" for r, q in EXAMPLES],
#         index=0,
#         help="Prefill query and route from a known-good example"
#     )
#     show_debug = st.checkbox("Show debug state", value=False)
#     st.markdown("---")
#     st.caption("History (latest 5)")
#     for item in st.session_state.history[-5:][::-1]:
#         st.write(f"- {item['route']} · {item['query'][:50]}… ({item['dt']:.2f}s)")

# # If user picked an example, update fields
# if ex_label != "(pick one)":
#     r, q = next((r, q) for r, q in EXAMPLES if f"{r} · {q}" == ex_label)
#     route_choice = r if r in ROUTES else "Auto (router picks)"
#     st.session_state["prefilled_query"] = q

# # ─────────────────────────── Main controls ───────────────────────────
# default_q = st.session_state.get(
#     "prefilled_query",
#     "Find a path between asthma and diabetes mellitus" if route_choice.endswith("pathfinder") else
#     "What proteins does acetaminophen interact with?"
# )

# query = st.text_input("Natural-language question", value=default_q)
# col1, col2 = st.columns([1, 1])
# run = col1.button("Run", use_container_width=True)
# clear = col2.button("Clear", use_container_width=True)

# if clear:
#     st.session_state.prefilled_query = ""
#     st.experimental_rerun()

# # ─────────────────────────── Helper: graph viz ───────────────────────────
# def render_qg_pyvis(qg: Dict[str, Any]) -> None:
#     if not _HAS_PYVIS:
#         st.info("Install `pyvis` to see a live network visualization: `pip install pyvis`.")
#         return
#     nodes = qg.get("nodes", {})
#     edges = qg.get("edges", {})
#     paths = qg.get("paths", {})

#     net = Network(height="420px", width="100%", directed=True, notebook=False)
#     net.barnes_hut()

#     # add nodes
#     for nid, meta in nodes.items():
#         label = meta.get("name") or ", ".join(meta.get("ids", []) or meta.get("categories", []) or [nid])
#         title = json.dumps(meta, indent=2)
#         net.add_node(nid, label=label, title=title)

#     # edges mode
#     if edges:
#         for eid, meta in edges.items():
#             subj, obj = meta.get("subject"), meta.get("object")
#             pred = ", ".join(meta.get("predicates", []))
#             net.add_edge(subj, obj, label=pred, title=json.dumps(meta, indent=2))

#     # pathfinder mode (paths.p0)
#     if paths and "p0" in paths:
#         p0 = paths["p0"]
#         subj, obj = p0.get("subject"), p0.get("object")
#         pred = ", ".join(p0.get("predicates", []))
#         net.add_edge(subj, obj, label=pred, title=json.dumps(p0, indent=2))

#     net.set_options("""{
#       "interaction": {"hover": true},
#       "physics": {"stabilization": true}
#     }""")
#     html = net.generate_html()
#     st.components.v1.html(html, height=440, scrolling=False)

# # ─────────────────────────── Run NL→TRAPI ───────────────────────────
# if run:
#     state: Dict[str, Any] = {"query": query}
#     if route_choice != "Auto (router picks)":
#         state["route"] = route_choice

#     with st.spinner("Running…"):
#         t0 = time.time()
#         final_state = graph.invoke(state)
#         dt = time.time() - t0

#     st.success(f"Done in {dt:.2f}s")

#     # Show status panel
#     st.subheader("Status")
#     colA, colB, colC = st.columns(3)
#     colA.metric("Route", final_state.get("route", "(unknown)"))
#     colB.metric("skip_schema", str(final_state.get("skip_schema")))
#     colC.metric("Valid", "Yes ✅" if final_state.get("valid") else "No ❌")

#     if final_state.get("errors"):
#         st.error("Validation errors:\n\n" + "\n".join(final_state["errors"]))

#     # TRAPI payload
#     st.subheader("TRAPI query_graph")
#     qg = (final_state or {}).get("output_json", {}).get("message", {}).get("query_graph", {})
#     full_payload = {"message": {"query_graph": qg}}
#     pretty = json.dumps(full_payload, indent=2)
#     st.code(pretty, language="json")

#     # Download button
#     st.download_button(
#         "Download JSON",
#         data=pretty.encode("utf-8"),
#         file_name="query_graph.trapi.json",
#         mime="application/json",
#         use_container_width=True,
#     )

#     # Graph visualization
#     st.subheader("Visualization")
#     render_qg_pyvis(qg)

#     # Optional debug
#     if show_debug:
#         st.subheader("Debug state (selected fields)")
#         debug = {
#             "route": final_state.get("route"),
#             "skip_schema": final_state.get("skip_schema"),
#             "entities": final_state.get("entities"),
#             "generic_types": final_state.get("generic_types"),
#             "predicate": final_state.get("predicate"),
#             "nodes_seen": list((final_state.get("nodes") or {}).keys()),
#             "errors": final_state.get("errors"),
#         }
#         st.json(debug)

#     # Save to history
#     st.session_state.history.append({"route": final_state.get("route"), "query": query, "dt": dt})

# # PYTHONPATH="$(pwd)" streamlit run scripts/demo_streamlit.py --server.port 7860 --server.address 0.0.0.0
# # lsof -i:7860
# # kill -9 3607435
# # pkill -f streamlit







#!/usr/bin/env python3
# Streamlit demo for NL2TRAPI-langgraph (pyvis + optional voice with lazy imports)

import json
import logging
import os
import time
import tempfile
from typing import Any, Dict, List, Tuple

import streamlit as st
from trapi_agent.agent_graph import graph

# ─────────────────────────── Optional deps (safe-guarded) ───────────────────────────
_HAS_PYVIS = False
try:
    from pyvis.network import Network  # pip install pyvis
    _HAS_PYVIS = True
except Exception:
    pass

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)

# ─────────────────────────── UI OPTIONS ───────────────────────────
ROUTES = [
    "Auto (router picks)",
    "onehop",
    "pathfinder",
    "treats",
    "chem_gene",
    "xcrg",
    "pathfinder_constrained",
]

EXAMPLES: List[Tuple[str, str]] = [
    ("onehop", "What proteins does acetaminophen interact with?"),
    ("onehop", "What biological processes are related to GFAP?"),
    ("onehop", "In which tissues is GFAP expressed?"),
    ("pathfinder", "Find a path between asthma and diabetes mellitus"),
    ("pathfinder", "By what paths are ibuprofen and headaches connected?"),
    ("treats", "What drugs treat asthma?"),
    ("treats", "What drugs treat diabetes mellitus?"),
    ("treats", "What drugs treat Castleman disease?"),
    ("treats", "What chemicals are predicted to be useful to treat malignant ciliary body melanoma?"),
    ("pathfinder", "How are neutropenia and filgrastim related in multi-hop paths?"),
    ("pathfinder", "Find me paths between ibuprofen and COX1"),
    ("xcrg", "What genes are upregulated by filgrastim?"),
    ("xcrg", "What genes are downregulated by filgrastim?"),
    ("xcrg", "Which drugs inhibit the activity of ABCB1?"),
    ("pathfinder_constrained", "Find a path between asthma and diabetes mellitus that includes a protein"),
    ("pathfinder_constrained", "Find a path between asthma and diabetes mellitus that includes a biological process"),
    ("pathfinder_constrained", "How are ibuprofen and headaches related via paths going through genes?"),
    ("pathfinder_constrained", "By what paths are BRCA1 and breast cancer connected via genes?"),
    ("pathfinder_constrained", "Find me paths between ibuprofen and COX1 via proteins"),
    ("pathfinder_constrained", "Find me paths between ibuprofen and COX1 via biological processes"),
    ("pathfinder_constrained", "Find me paths between ibuprofen and COX1 via diseases"),
    ("pathfinder_constrained", "How are EGFR and lung cancer related through proteins?"),
    ("pathfinder_constrained", "How are neutropenia and filgrastim related in multi-hop paths going through diseases?"),
    ("pathfinder_constrained", "How are obesity and insulin resistance connected via diseases?"),
    ("pathfinder_constrained", "Show connections between BRCA1 and asthma via a drug"),
    ("pathfinder_constrained", "Paths between TNF and rheumatoid arthritis via a chemical"),
    ("pathfinder_constrained", "Find paths from LRRK2 to Parkinson disease via small molecules"),
    ("pathfinder_constrained", "Show multi-hop paths between BRCA1 and DNA repair through pathways"),
    ("pathfinder_constrained", "Show paths from kinase inhibitors to EGFR via molecular activities"),
    ("pathfinder_constrained", "How are COX1 and prostaglandin synthesis related through activities?"),
    ("pathfinder_constrained", "Find paths between GFAP and seizures via tissues"),
    ("pathfinder_constrained", "Show paths between HIF1A and hypoxia via cells"),
    ("pathfinder_constrained", "How are BRCA1 and DNA repair related through organelles?"),
    ("pathfinder_constrained", "Show paths between APOE and Alzheimer disease through phenotypes"),
    ("pathfinder_constrained", "Find paths from TP53 to cancer via phenotypes"),
]

st.set_page_config(page_title="NL → TRAPI Demo", layout="wide")
st.title("🧠 NL → TRAPI Demo")

# Minimal session state
if "history" not in st.session_state:
    st.session_state.history = []
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# ─────────────────────────── Sidebar ───────────────────────────
with st.sidebar:
    st.header("Settings")
    route_choice = st.selectbox("Route", ROUTES, index=0)

    ex_label = st.selectbox(
        "Examples",
        ["(pick one)"] + [f"{r} · {q}" for r, q in EXAMPLES],
        index=0,
        help="Prefill query and route from a known-good example",
    )

    show_debug = st.checkbox("Show debug state", value=False)
    st.markdown("---")
    enable_voice = st.checkbox("Enable voice input (beta, CPU by default)", value=False,
                               help="Lazy-loads deps only when ON. Avoids CUDA/cuDNN issues.")

    st.caption("History (latest 5)")
    for item in st.session_state.history[-5:][::-1]:
        st.write(f"- {item['route']} · {item['query'][:50]}… ({item['dt']:.2f}s)")

# Apply example pick
if ex_label != "(pick one)":
    r, q = next((r, q) for r, q in EXAMPLES if f"{r} · {q}" == ex_label)
    route_choice = r if r in ROUTES else "Auto (router picks)"
    st.session_state["prefilled_query"] = q
    st.session_state["query_text"] = q

# ─────────────────────────── Optional: Voice → Text (lazy import) ────────────────────
st.divider()
st.write("🎙️ **Voice input (local, open-source)**")

def _lazy_import_audio_recorder():
    import importlib
    return importlib.import_module("audio_recorder_streamlit").audio_recorder  # pip install audio-recorder-streamlit

@st.cache_resource
def _load_whisper(device: str = "cpu"):
    """
    device: "cpu" (safe default) or "cuda"
    """
    import importlib
    WhisperModel = importlib.import_module("faster_whisper").WhisperModel  # pip install faster-whisper ctranslate2
    if device == "cuda":
        try:
            return WhisperModel(
                "small.en",
                device="cuda",
                device_index=int(os.getenv("WHISPER_DEVICE_INDEX", "0")),
                compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
                num_workers=1,
            )
        except Exception as e:
            st.warning(f"Whisper GPU unavailable ({e}). Falling back to CPU.")
    # CPU path (no CUDA/cuDNN needed)
    return WhisperModel("small.en", device="cpu", compute_type="int8")

if enable_voice:
    try:
        audio_recorder = _lazy_import_audio_recorder()
        # Tip: set CT2_USE_CUDA=0 to ensure ctranslate2 stays on CPU even if CUDA is present
        os.environ.setdefault("CT2_USE_CUDA", "0")  # keep Whisper on CPU by default
        audio_bytes = audio_recorder(
            text="Click to record • click again to stop",
            sample_rate=16_000,
            pause_threshold=2.0,
            icon_size="2x",
        )
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                wav_path = f.name

            model = _load_whisper(device=os.getenv("VOICE_DEVICE", "cpu"))  # set VOICE_DEVICE=cuda only if cuDNN is good
            # Avoid VAD to dodge possible media/ML backends
            segments, _ = model.transcribe(
                wav_path,
                vad_filter=False,
                beam_size=1,
                language="en",
            )
            text = "".join(seg.text for seg in segments).strip()
            if text:
                st.session_state["query_text"] = text
                st.success(f"Transcribed: “{text}” — click **Run** below.")
            else:
                st.warning("No speech detected—try again closer to the mic.")
    except ModuleNotFoundError:
        st.info("To enable voice: `pip install audio-recorder-streamlit faster-whisper ctranslate2`")
    except Exception as e:
        st.error(f"Voice input failed safely: {e}")

# ─────────────────────────── Main controls ───────────────────────────
default_q = (
    st.session_state.get("query_text")
    or st.session_state.get("prefilled_query")
    or ("Find a path between asthma and diabetes mellitus" if route_choice.endswith("pathfinder") else
        "What proteins does acetaminophen interact with?")
)

query = st.text_input("Natural-language question", value=default_q, key="query_text")
col1, col2, col3 = st.columns([1, 1, 1])
run = col1.button("Run", use_container_width=True)
clear = col2.button("Clear", use_container_width=True)
clear_hist = col3.button("Clear history", use_container_width=True)

if clear:
    st.session_state.prefilled_query = ""
    st.session_state.query_text = ""
    st.rerun()

if clear_hist:
    st.session_state.history = []
    st.rerun()

# ─────────────────────────── Helper: graph viz ───────────────────────────
def render_qg_pyvis(qg: Dict[str, Any]) -> None:
    if not _HAS_PYVIS:
        st.info("Install `pyvis` to see a live network visualization: `pip install pyvis`.")
        return
    nodes = qg.get("nodes", {})
    edges = qg.get("edges", {})
    paths = qg.get("paths", {})

    net = Network(height="420px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    # nodes
    for nid, meta in nodes.items():
        label = meta.get("name") or ", ".join(
            meta.get("ids", []) or meta.get("categories", []) or [nid]
        )
        net.add_node(nid, label=label, title=json.dumps(meta, indent=2))

    # edges
    if edges:
        for _, meta in edges.items():
            subj, obj = meta.get("subject"), meta.get("object")
            pred = ", ".join(meta.get("predicates", []))
            net.add_edge(subj, obj, label=pred, title=json.dumps(meta, indent=2))

    # pathfinder "paths"
    if paths and "p0" in paths:
        p0 = paths["p0"]
        subj, obj = p0.get("subject"), p0.get("object")
        pred = ", ".join(p0.get("predicates", []))
        net.add_edge(subj, obj, label=pred, title=json.dumps(p0, indent=2))

    net.set_options("""{
      "interaction": {"hover": true},
      "physics": {"stabilization": true}
    }""")
    html = net.generate_html()
    st.components.v1.html(html, height=440, scrolling=False)

# ─────────────────────────── Run NL→TRAPI ───────────────────────────
if run:
    state: Dict[str, Any] = {"query": query}
    if route_choice != "Auto (router picks)":
        state["route"] = route_choice

    with st.spinner("Running…"):
        t0 = time.time()
        final_state = graph.invoke(state)
        dt = time.time() - t0

    st.success(f"Done in {dt:.2f}s")

    # Status
    st.subheader("Status")
    colA, colB, colC = st.columns(3)
    colA.metric("Route", final_state.get("route", "(unknown)"))
    colB.metric("skip_schema", str(final_state.get("skip_schema")))
    colC.metric("Valid", "Yes ✅" if final_state.get("valid") else "No ❌")

    if final_state.get("errors"):
        st.error("Validation errors:\n\n" + "\n".join(final_state["errors"]))

    # TRAPI payload
    st.subheader("TRAPI query_graph")
    qg = (final_state or {}).get("output_json", {}).get("message", {}).get("query_graph", {})
    st.code(json.dumps({"message": {"query_graph": qg}}, indent=2), language="json")

    # Download
    st.download_button(
        "Download JSON",
        data=json.dumps({"message": {"query_graph": qg}}, indent=2).encode("utf-8"),
        file_name="query_graph.trapi.json",
        mime="application/json",
        use_container_width=True,
    )

    # Viz
    st.subheader("Visualization")
    render_qg_pyvis(qg)

    # Debug
    if show_debug:
        st.subheader("Debug state (selected fields)")
        st.json({
            "route": final_state.get("route"),
            "skip_schema": final_state.get("skip_schema"),
            "entities": final_state.get("entities"),
            "generic_types": final_state.get("generic_types"),
            "predicate": final_state.get("predicate"),
            "nodes_seen": list((final_state.get("nodes") or {}).keys()),
            "errors": final_state.get("errors"),
        })

    # History
    st.session_state.history.append({"route": final_state.get("route"), "query": query, "dt": dt})
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-50:]  # cap history size


# # PYTHONPATH="$(pwd)" streamlit run scripts/demo_streamlit.py --server.port 7860 --server.address 0.0.0.0
# # lsof -i:7860
# # kill -9 3607435
# # pkill -f streamlit