import chromadb
from chromadb.config import Settings
import json

def initialize_chroma_db():
    return chromadb.PersistentClient(path="/data/vmm5481/nl_to_trapi_db") # adjust the path accordingly

def add_to_collection_if_not_exists(collection, documents, metadatas, ids):
    existing_data = collection.get(ids=ids)
    existing_ids = set(existing_data['ids']) if 'ids' in existing_data else set()
    
    new_ids = set(ids)
    ids_to_add = list(new_ids - existing_ids)
    
    if ids_to_add:
        collection.add(
            documents=[documents[i] for i in range(len(ids)) if ids[i] in ids_to_add],
            metadatas=[metadatas[i] for i in range(len(ids)) if ids[i] in ids_to_add],
            ids=ids_to_add
        )
    else:
        print(f"Skipping duplicate IDs: {existing_ids.intersection(new_ids)}")

def populate_nl_to_trapi_examples(chroma_client, collection_name):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    nl_to_trapi_examples = [
        # Example 1
        {
            "nl_query": "What proteins does acetaminophen interact with?",
            "trapi_query": {
                "edges": {
                    "qg2": {
                        "subject": "qg1", 
                        "object": "qg0", 
                        "predicates": ["biolink:physically_interacts_with"]
                    }
                },
                "nodes": {
                    "qg0": {
                        "name": "acetaminophen",
                        "ids": ["CHEMBL.COMPOUND:CHEMBL112"],
                        "categories": ["biolink:ChemicalEntity"]
                    },
                    "qg1": {"categories": ["biolink:Protein"]}
                }
            }
        },
        
        # Example 2
        {
            "nl_query": "What drugs are both indicated and contraindicated for bipolar disorder?",
            "trapi_query": {
                "edges": {
                    "e00": {
                        "subject": "n00",
                        "object": "n01",
                        "predicates": ["biolink:treats_or_applied_or_studied_to_treat"]
                    },
                    "e01": {
                        "subject": "n00",
                        "object": "n01", 
                        "predicates": ["biolink:contraindicated_in"]
                    }
                },
                "nodes": {
                    "n00": {"ids": ["DOID:3312"]},
                    "n01": {"categories": ["biolink:ChemicalEntity"]}
                }
            }
        },

        # Example 3
        {
            "nl_query": "What chemicals treat bipolar disorder, and are not contraindicated for it",
            "trapi_query": {
                "edges": {
                    "e00": {"exclude": False, "object": "n01", "predicates": ["biolink:treats"], "subject": "n00"},
                    "e01": {"exclude": True, "object": "n01", "predicates": ["biolink:contraindicated_for"], "subject": "n00"},
                    "subclass:n00--n00": {"object": "n00", "predicates": ["biolink:subclass_of"], "subject": "n00"}
                },
                "nodes": {
                    "n00": {"ids": ["MONDO:0004985"]},
                    "n01": {"categories": ["biolink:ChemicalEntity"]}
                }
            }
        },

        # Example 4
        {
            "nl_query": "What are chemicals that interact with genes that are connected to gonorrhea?",
            "trapi_query": {
                "edges": {
                    "e0": {"subject": "n0", "object": "n1"},
                    "e1": {"subject": "n1", "object": "n2", "predicates": ["biolink:physically_interacts_with"]}
                },
                "nodes": {
                    "n0": {"ids": ["MONDO:0004277"], "categories": ["biolink:Disease"], "name": "gonorrhea"},
                    "n1": {"categories": ["biolink:Protein"]},
                    "n2": {"categories": ["biolink:ChemicalEntity"]}
                }
            }
        },

        # Example 5
        # {
        #     "nl_query": "What biological processes are related to GFAP",
        #     "trapi_query": {
        #         "edges": {"e00": {"subject": "n00", "object": "n01"}},
        #         "nodes": {
        #             "n00": {"categories": ["biolink:Protein"], "ids": ["UniProtKB:P14136"]},
        #             "n01": {"categories": ["biolink:BiologicalProcess"]}
        #         }
        #     }
        # },

        # Example 6
        {
            "nl_query": "What genes are related to myopia and what are chemicals that are related to those genes",
            "trapi_query": {
                "edges": {
                    "e0": {"subject": "n0", "object": "n1"},
                    "e1": {"subject": "n1", "object": "n2"}
                },
                "nodes": {
                    "n0": {"ids": ["MONDO:0001384"]},
                    "n1": {"categories": ["biolink:Gene"]},
                    "n2": {"categories": ["biolink:ChemicalEntity"]}
                }
            }
        },

        # Example 7
        {
            "nl_query": "How are aldehydo-L-glucose and diabetes related",
            "trapi_query": {
                "nodes": {
                    "n0": {"ids": ["CHEBI:37626"]},
                    "un": {"categories": ["biolink:NamedThing"]},
                    "n2": {"ids": ["MONDO:0005015"]}
                },
                "edges": {
                    "e0": {"subject": "n0", "object": "un", "predicates": ["biolink:related_to"], "knowledge_type": "inferred"},
                    "e1": {"subject": "un", "object": "n2", "predicates": ["biolink:related_to"], "knowledge_type": "inferred"},
                    "e2": {"subject": "n0", "object": "n2", "predicates": ["biolink:related_to"], "knowledge_type": "inferred"}
                }
            }
        },

        # Example 8
        {
            "nl_query": "What are some drugs that might be repurposed to treat castleman's disease?",
            "trapi_query": {
                "edges": {
                    "e01": {"subject": "SN", "object": "ON", "predicates": ["biolink:treats"], "knowledge_type": "inferred"}
                },
                "nodes": {
                    "ON": {"categories": ["biolink:Disease"], "ids": ["MONDO:0015564"]},
                    "SN": {"categories": ["biolink:ChemicalEntity"]}
                }
            }
        },

        # Example 9
        {
            "nl_query": "What drugs affect genes that are expressed in tissues related to cystic fibrosis",
            "trapi_query": {
                "nodes": {
                    "n0": {"categories": ["biolink:Disease"], "ids": ["MONDO:0009061"]},
                    "n1": {"categories": ["biolink:GrossAnatomicalStructure"]},
                    "n2": {"categories": ["biolink:Gene"]},
                    "n3": {"categories": ["biolink:Drug", "biolink:SmallMolecule"]}
                },
                "edges": {
                    "e0": {"subject": "n0", "object": "n1", "predicates": ["biolink:related_to"]},
                    "e1": {"subject": "n1", "object": "n2", "predicates": ["biolink:expresses"]},
                    "e2": {"subject": "n3", "object": "n2", "predicates": ["biolink:affects"]}
                }
            }
        },

        # Example 10
        {
            "nl_query": "What are symptoms of tetralogy of fallot, gastric adenocarcinoma, and choroiditis?",
            "trapi_query": {
                "edges": {
                    "e00": {"subject": "n00", "object": "n01", "predicates": ["biolink:has_phenotype"]},
                    "subclass:n00--n00": {"subject": "n00", "object": "n00", "predicates": ["biolink:subclass_of"]}
                },
                "nodes": {
                    "n00": {"ids": ["DOID:6419", "DOID:3717", "DOID:11406"]},
                    "n01": {"categories": ["biolink:PhenotypicFeature"]}
                }
            }
        },

        # Example 11
        {
            "nl_query": "What are genes that decrease the activity of the gene TP53",
            "trapi_query": {
                "edges": {
                    "t_edge": {
                        "subject": "sn",
                        "object": "on",
                        "predicates": ["biolink:affects"],
                        "qualifier_constraints": [{
                            "qualifier_set": [
                                {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity_or_abundance"},
                                {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "decreased"}
                            ]
                        }],
                        "knowledge_type": "inferred"
                    }
                },
                "nodes": {
                    "on": {"categories": ["biolink:Gene"], "ids": ["NCBIGene:7157"]},
                    "sn": {"categories": ["biolink:ChemicalEntity"]}
                }
            }
        },

        # Example 12
        {
            "nl_query": "What are chemicals related to tyrosinemia type 1, and what are genes related to those chemicals and that disease?",
            "trapi_query": {
                "edges": {
                    "e0": {"subject": "n1", "object": "n0", "predicates": ["biolink:related_to"]},
                    "e1": {"subject": "n1", "object": "n2", "predicates": ["biolink:related_to"]},
                    "e2": {"subject": "n0", "object": "n2", "predicates": ["biolink:related_to"]}
                },
                "nodes": {
                    "n0": {"ids": ["MONDO:0010161"]},
                    "n1": {"categories": ["biolink:Gene"]},
                    "n2": {"categories": ["biolink:ChemicalEntity"]}
                }
            }
        }
    ]

    for example in nl_to_trapi_examples:
        nl_query = example['nl_query']
        trapi_query = json.dumps(example['trapi_query'])
        
        add_to_collection_if_not_exists(
            collection=collection,
            documents=[nl_query],
            metadatas=[{"trapi_query": trapi_query}],
            ids=[nl_query]
        )

def main():
    nl_to_trapi_db = initialize_chroma_db()
    populate_nl_to_trapi_examples(nl_to_trapi_db, "nl_to_trapi")
    print("Successfully loaded 12 NL-to-TRAPI examples into ChromaDB (DB1)")

if __name__ == "__main__":
    main()
