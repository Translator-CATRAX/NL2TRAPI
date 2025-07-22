import spacy, json

nlp = spacy.load("en_ner_bionlp13cg_md")
nlp.disable_pipes("tagger", "parser", "attribute_ruler", "lemmatizer")

doc = nlp("What proteins does acetaminophen interact with?")
print(json.dumps([(ent.text, ent.label_) for ent in doc.ents], indent=2))
