#!/usr/bin/env python3
# Streamlit demo for NL2TRAPI-langgraph (enhanced)

import json
import logging
import time
from io import StringIO
from typing import Any, Dict, List, Tuple

import streamlit as st

# Graph viz (optional)
try:
    from pyvis.network import Network  # pip install pyvis
    _HAS_PYVIS = True
except Exception:
    _HAS_PYVIS = False

from trapi_agent.agent_graph import graph

# ─────────────────────────── UI OPTIONS ───────────────────────────
ROUTES = ["Auto (router picks)", "onehop", "pathfinder", "treats", "chem_gene"]

EXAMPLES: List[Tuple[str, str]] = [
    ("onehop", "What proteins does acetaminophen interact with?"),
    ("onehop", "What biological processes are related to GFAP?"),
    ("pathfinder", "Find a path between asthma and diabetes mellitus"),
    ("pathfinder", "By what paths are ibuprofen and headaches connected?"),
    ("treats", "What drugs treat asthma?"),
]

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)

st.set_page_config(page_title="NL → TRAPI Demo", layout="wide")
st.title("🧠 NL → TRAPI Demo")

# minimal session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts

# ─────────────────────────── Sidebar ───────────────────────────
with st.sidebar:
    st.header("Settings")
    route_choice = st.selectbox("Route", ROUTES, index=0)
    ex_label = st.selectbox(
        "Examples",
        ["(pick one)"] + [f"{r} · {q}" for r, q in EXAMPLES],
        index=0,
        help="Prefill query and route from a known-good example"
    )
    show_debug = st.checkbox("Show debug state", value=False)
    st.markdown("---")
    st.caption("History (latest 5)")
    for item in st.session_state.history[-5:][::-1]:
        st.write(f"- {item['route']} · {item['query'][:50]}… ({item['dt']:.2f}s)")

# If user picked an example, update fields
if ex_label != "(pick one)":
    r, q = next((r, q) for r, q in EXAMPLES if f"{r} · {q}" == ex_label)
    route_choice = r if r in ROUTES else "Auto (router picks)"
    st.session_state["prefilled_query"] = q

# ─────────────────────────── Main controls ───────────────────────────
default_q = st.session_state.get(
    "prefilled_query",
    "Find a path between asthma and diabetes mellitus" if route_choice.endswith("pathfinder") else
    "What proteins does acetaminophen interact with?"
)

query = st.text_input("Natural-language question", value=default_q)
col1, col2 = st.columns([1, 1])
run = col1.button("Run", use_container_width=True)
clear = col2.button("Clear", use_container_width=True)

if clear:
    st.session_state.prefilled_query = ""
    st.experimental_rerun()

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

    # add nodes
    for nid, meta in nodes.items():
        label = meta.get("name") or ", ".join(meta.get("ids", []) or meta.get("categories", []) or [nid])
        title = json.dumps(meta, indent=2)
        net.add_node(nid, label=label, title=title)

    # edges mode
    if edges:
        for eid, meta in edges.items():
            subj, obj = meta.get("subject"), meta.get("object")
            pred = ", ".join(meta.get("predicates", []))
            net.add_edge(subj, obj, label=pred, title=json.dumps(meta, indent=2))

    # pathfinder mode (paths.p0)
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

    # Show status panel
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
    full_payload = {"message": {"query_graph": qg}}
    pretty = json.dumps(full_payload, indent=2)
    st.code(pretty, language="json")

    # Download button
    st.download_button(
        "Download JSON",
        data=pretty.encode("utf-8"),
        file_name="query_graph.trapi.json",
        mime="application/json",
        use_container_width=True,
    )

    # Graph visualization
    st.subheader("Visualization")
    render_qg_pyvis(qg)

    # Optional debug
    if show_debug:
        st.subheader("Debug state (selected fields)")
        debug = {
            "route": final_state.get("route"),
            "skip_schema": final_state.get("skip_schema"),
            "entities": final_state.get("entities"),
            "generic_types": final_state.get("generic_types"),
            "predicate": final_state.get("predicate"),
            "nodes_seen": list((final_state.get("nodes") or {}).keys()),
            "errors": final_state.get("errors"),
        }
        st.json(debug)

    # Save to history
    st.session_state.history.append({"route": final_state.get("route"), "query": query, "dt": dt})

# PYTHONPATH="$(pwd)" streamlit run scripts/demo_streamlit.py --server.port 7860 --server.address 0.0.0.011