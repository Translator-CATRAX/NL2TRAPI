# import spacy, json

# nlp = spacy.load("en_ner_bionlp13cg_md")
# nlp.disable_pipes("tagger", "parser", "attribute_ruler", "lemmatizer")

# doc = nlp("What proteins does acetaminophen interact with?")
# print(json.dumps([(ent.text, ent.label_) for ent in doc.ents], indent=2))

# import spacy, json

# EXAMPLES = [
#     ("onehop", "What proteins does acetaminophen interact with?"),
#     ("onehop", "What biological processes are related to GFAP?"),
#     ("onehop", "In which tissues is GFAP expressed?"),
#     ("pathfinder", "Find a path between asthma and diabetes mellitus"),
#     ("pathfinder", "By what paths are ibuprofen and headaches connected?"),
#     ("treats", "What drugs treat asthma?"),
#     ("treats", "What drugs treat diabetes mellitus?"),
#     ("treats", "What drugs treat Castleman disease?"),
#     ("treats", "What chemicals are predicted to be useful to treat malignant ciliary body melanoma?"),
#     ("pathfinder", "How are neutropenia and filgrastim related in multi-hop paths?"),
#     ("pathfinder", "Find me paths between ibuprofen and COX1"),
#     ("xcrg", "What genes are upregulated by filgrastim?"),
#     ("xcrg", "What genes are downregulated by filgrastim?"),
#     ("xcrg", "Which drugs inhibit the activity of ABCB1?"),
#     ("pathfinder_constrained", "Find a path between asthma and diabetes mellitus that includes a protein"),
#     ("pathfinder_constrained", "Find a path between asthma and diabetes mellitus that includes a biological process"),
#     ("pathfinder_constrained", "How are ibuprofen and headaches related via paths going through genes?"),
#     ("pathfinder_constrained", "By what paths are BRCA1 and breast cancer connected via genes?"),
#     ("pathfinder_constrained", "Find me paths between ibuprofen and COX1 via proteins"),
#     ("pathfinder_constrained", "Find me paths between ibuprofen and COX1 via biological processes"),
#     ("pathfinder_constrained", "Find me paths between ibuprofen and COX1 via diseases"),
#     ("pathfinder_constrained", "How are EGFR and lung cancer related through proteins?"),
#     ("pathfinder_constrained", "How are neutropenia and filgrastim related in multi-hop paths going through diseases?"),
#     ("pathfinder_constrained", "How are obesity and insulin resistance connected via diseases?"),
#     ("pathfinder_constrained", "Show connections between BRCA1 and asthma via a drug"),
#     ("pathfinder_constrained", "Paths between TNF and rheumatoid arthritis via a chemical"),
#     ("pathfinder_constrained", "Find paths from LRRK2 to Parkinson disease via small molecules"),
#     ("pathfinder_constrained", "Show multi-hop paths between BRCA1 and DNA repair through pathways"),
#     ("pathfinder_constrained", "Show paths from kinase inhibitors to EGFR via molecular activities"),
#     ("pathfinder_constrained", "How are COX1 and prostaglandin synthesis related through activities?"),
#     ("pathfinder_constrained", "Find paths between GFAP and seizures via tissues"),
#     ("pathfinder_constrained", "Show paths between HIF1A and hypoxia via cells"),
#     ("pathfinder_constrained", "How are BRCA1 and DNA repair related through organelles?"),
#     ("pathfinder_constrained", "Show paths between APOE and Alzheimer disease through phenotypes"),
#     ("pathfinder_constrained", "Find paths from TP53 to cancer via phenotypes"),
# ]

# nlp = spacy.load("en_ner_bionlp13cg_md")
# nlp.disable_pipes("tagger", "parser", "attribute_ruler", "lemmatizer")

# for route, text in EXAMPLES:
#     doc = nlp(text)
#     ents = [(ent.text, ent.label_) for ent in doc.ents]
#     print(f"\n=== {route} | {text}")
#     print(json.dumps(ents, indent=2))



from trapi_agent.nodes import parse_query, resolve_entities
from trapi_agent.state_types import TRAPIState

def debug(query: str, route: str):
    state: TRAPIState = {"query": query, "route": route}
    state = parse_query.node(state)
    state = resolve_entities.node(state)
    print("query:", query)
    print("entities:", state.get("entities"))
    print("nodes:", state.get("nodes"))
    print()

debug("By what paths are ibuprofen and headaches connected?", "pathfinder")
debug("What drugs treat asthma?", "treats")
debug("What drugs treat diabetes mellitus?", "treats")
debug("What drugs treat Castleman disease?", "treats")
debug("Find a path between asthma and diabetes mellitus that includes a protein", "pathfinder_constrained")
debug("What proteins does acetaminophen interact with?", "onehop")
debug("What biological processes are related to GFAP?","onehop")
debug("In which tissues is GFAP expressed?","onehop")