#!/usr/bin/env python3
from __future__ import annotations

import json
import gradio as gr
from trapi_agent.agent_graph import graph

ROUTES = ["Auto", "onehop", "pathfinder", "treats", "chem_gene"]

def run_agent(query: str, route: str):
    if not query.strip():
        return "Please enter a query.", "", "", ""
    state = {"query": query.strip()}
    if route != "Auto":
        state["route"] = route
    fs = graph.invoke(state)
    trapi = json.dumps(fs.get("output_json", {}), indent=2)
    details = {
        "entities": fs.get("entities"),
        "ner_spans": fs.get("ner_spans"),
        "generic_types": fs.get("generic_types"),
        "predicate": fs.get("predicate"),
        "nodes": fs.get("nodes"),
        "edges": fs.get("edges"),
        "valid": fs.get("valid"),
        "errors": fs.get("errors"),
        "route": fs.get("route"),
        "skip_schema": fs.get("skip_schema"),
    }
    return ("✅ Valid" if fs.get("valid") else "⚠️ Invalid"), trapi, json.dumps(details, indent=2), route

with gr.Blocks(title="NL→TRAPI Demo") as demo:
    gr.Markdown("# 🧠 NL→TRAPI Demo")
    with gr.Row():
        query = gr.Textbox(label="Natural-language query", lines=3,
                           placeholder="Find a path between asthma and diabetes mellitus")
        route = gr.Dropdown(ROUTES, value="Auto", label="Route")
    run_btn = gr.Button("Run 🔍")
    status = gr.Textbox(label="Status", interactive=False)
    trapi_json = gr.Code(label="TRAPI output_json", language="json")
    debug_json = gr.Code(label="Debug (parsed/resolved state)", language="json")
    _chosen = gr.Textbox(visible=False)

    run_btn.click(run_agent, inputs=[query, route], outputs=[status, trapi_json, debug_json, _chosen])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
