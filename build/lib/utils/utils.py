from SPARQLWrapper import SPARQLWrapper, JSON

# 1) Definir endpoint
endpoint_url = "http://dbpedia.org/sparql"

# 2) Consulta sencilla: cinco tríos cualesquiera
query = """
SELECT ?s ?p ?o
WHERE {
  ?s ?p ?o .
}
LIMIT 5
"""

# 3) Ejecutar con SPARQLWrapper
sparql = SPARQLWrapper(endpoint_url)
sparql.setQuery(query)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

# 4) Mostrar resultados
for binding in results["results"]["bindings"]:
    s = binding["s"]["value"]
    p = binding["p"]["value"]
    o = binding["o"]["value"]
    print(f"{s} → {p} → {o}")
