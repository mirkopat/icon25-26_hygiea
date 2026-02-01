"""
Integrazione conoscenza esterna - Web Semantico
Linea-guida: "integrazione con conoscenza di fondo dal Web Semantico"
"""
import requests
import json
from pathlib import Path

def fetch_disease_info(disease_name):
    """Recupera info da DBpedia (Knowledge Graph esterno)"""
    # Query SPARQL migliorata
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    
    SELECT ?desc WHERE {{
      ?disease rdfs:label ?label .
      ?disease dbo:abstract ?desc .
      FILTER (CONTAINS(LCASE(?label), "{disease_name.lower()}") && LANG(?desc) = 'en')
    }} LIMIT 1
    """
    
    url = "https://dbpedia.org/sparql"
    params = {
        'query': query,
        'format': 'json'
    }
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['results']['bindings']:
                desc = data['results']['bindings'][0]['desc']['value']
                return desc[:300] + "..."
            else:
                return f"Informazioni su {disease_name} non trovate in DBpedia"
        else:
            return f"Errore server: {response.status_code}"
        
    except requests.exceptions.Timeout:
        return "Timeout: DBpedia non risponde"
    except requests.exceptions.RequestException as e:
        return f"Errore connessione: {str(e)}"
    except Exception as e:
        return f"Errore generico: {str(e)}"

# Esempio d'uso
if __name__ == "__main__":
    diseases = ['Diabetes', 'Hypertension', 'Depression']
    for d in diseases:
        info = fetch_disease_info(d)
        print(f"{d}: {info}")