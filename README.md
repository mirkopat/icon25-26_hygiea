# Hygiea - Sistema di Prevenzione di Patologie Basato su Conoscenza

Sistema intelligente per l’analisi del rischio e la prevenzione personalizzata di malattie croniche, che integra **Machine Learning**, **Ragionamento Probabilistico** e **Rappresentazione della Conoscenza**.

---

**Progetto per l’esame di Ingegneria della Conoscenza**  
A.A. 2025/2026 – Università degli Studi di Bari

- **Autore:** Patruno Mirko (mat. 797729)  

## 📋 Indice
- [Panoramica del Sistema](#panoramica-del-sistema)
- [Installazione Rapida](#installazione-rapida)
- [Struttura del Progetto](#struttura-del-progetto)
- [Esecuzione del Sistema](#esecuzione-del-sistema)
- [Analisi di Complessità](#-analisi-di-complessità)
- [Risultati e Valutazione](#-risultati-e-valutazione)
- [Disclaimer](#-disclaimer)
- [Licenza](#-licenza)


---

##  Panoramica del Sistema

L'idea alla base del progetto si focalizza sulla prevenzione di patologie su diversi profili. **Hygiea** (dal nome Igea, o Hygieia, dea greca della salute, dell'igiene e della prevenzione delle malattie) è un sistema basato su conoscenza che integra **quattro modelli fondamentali di Intelligenza Artificiale** per la prevenzione personalizzata di malattie croniche.

### 🎯 Obiettivi Didattici Raggiunti
-  Integrazione multi-modello: **Markov + ML + CSP + Knowledge Base**
-  Valutazione robusta con metriche statistiche complete
-  Knowledge Base dichiarativa in **Prolog** con inferenza avanzata
-  Ottimizzazione vincolata tramite **CSP** e algoritmi di consistenza

### 🩺 Patologie Analizzate
1. **Depressione** (neurologica)
2. **Diabete** (metabolica)
3. **Ipertensione** (cardiovascolare)

---

##  Installazione Rapida

### Prerequisiti
- Python **3.8+** (testato su 3.10)
- **SWI-Prolog** (per la Knowledge Base)
- 2 GB RAM
- 200 MB spazio disco
- Un po’ di pazienza 🙂

### Procedura

#### 1️⃣ Clona il repository
```bash
git clone https://github.com/mirkopat/icon25-26_hygiea.git
cd icon25-26_hygiea
```
2️⃣ Crea e attiva un ambiente virtuale
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```
3️⃣ Installa le dipendenze
```bash
pip install -r requirements.txt
```
Dipendenza chiave: pyswip>=0.2.10 per l’integrazione Python–Prolog

### Struttura del Progetto
```text
icon25-26_hygiea/
├── .gitattributes
├── LICENSE
├── README.md
├── requirements.txt
│
├── docs/
│   └── immagini/
│       ├── classification_complete.png
│       └── markov_complete_analysis.png
│
├── src/
│   ├── 01_create_dataset.py        # Generazione dataset sintetico
│   ├── 02_markov_model.py          # Modello a Catene di Markov
│   ├── 03_classification.py        # Modelli di classificazione ML
│   ├── 04_csp_recommender.py       # CSP per raccomandazioni
│   ├── 05_integration.py           # Integrazione dei moduli
│   ├── 06_create_images.py         # Generazione immagini/plot
│   ├── external_knowledge.py       # Query DBpedia / Web Semantico
│   ├── integration_demo.py         # Demo del sistema integrato
│   ├── main_system.py              # Entry point del sistema
│   ├── wellness_kb.pl              # Knowledge Base Prolog
│   │
│   ├── data/
│   │   ├── simple_dataset.csv
│   │   ├── classification_results.csv
│   │   ├── csp_solution.json
│   │   ├── integrated_analysis.json
│   │   ├── final_report.json
│   │   └── wellness_kb.json
│   │
│   ├── venv/                       # Ambiente virtuale (non versionare)
│   │   ├── pyvenv.cfg
│   │   ├── Include/
│   │   ├── Lib/
│   │   │   └── site-packages/
│   │   └── Scripts/
│   │       ├── python.exe
│   │       └── pythonw.exe
│   │
│   └── __pycache__/
│       ├── external_knowledge.cpython-310.pyc
│       └── integration_demo.cpython-310.pyc
```
### Esecuzione del Sistema
Opzione 1 – Esecuzione completa
```bash
python src/integration_demo.py
```
Esegue l’intera pipeline su un paziente di esempio.

Opzione 2 – Esecuzione modulare
```yaml
Modulo	        Comando	                              Descrizione	                        Tempo
Dataset	        python src/dataset_generator.py	      Genera 60 pazienti simulati           30 s
Markov	        python src/markov_analyzer.py	        Analisi transizioni di rischio      1 m
ML	            python src/classification_models.py	  Classificazione 3 patologie           2 m
CSP	            python src/csp_solver.py	            Ottimizzazione interventi           30 s
Integrazione	 python src/integration_demo.py	        Sistema completo                    3 m
```
Opzione 3 – Test Knowledge Base 
```prolog
?- consult('src/wellness_kb.pl').

?- rischio_alto(marco, X).
?- punteggio_rischio(anna, diabete, P).
?- raccomanda_intervento(luigi, I).
?- spiega_rischio(maria, depressione, F).
```
## 🔍 Analisi di Complessità
1️⃣ Knowledge Base Prolog
Fatti: 15

Regole: 12 (3 abduttive)

Profondità inferenza: 3
```yaml
Query	                      Complessità
rischio_alto/2	              O(n²)
punteggio_rischio/3	          O(n)
raccomanda_intervento/2	      O(n²)
```
2️⃣ Catene di Markov
Matrici 3×3 per 3 patologie

Distribuzione stazionaria: O(k³)

Tempo di mixing: O(log(1/ε))

Simulazione: O(t)

3️⃣ Apprendimento Supervisionato
```yaml
Modello	                    Complessità
Random Forest	              O(m · n log n)
SVM (RBF)	                  O(n² · m)
Decision Tree	              O(m · n²)
```
4️⃣ CSP Solver
- AC-3: O(e · d³)
- Backtracking: O(dⁿ) (ridotto con pruning)

5️⃣ Sistema Integrato
- Complessità totale: O(max(n², k³, m·n log n))
- Tempo esecuzione: ~5 minuti

## 📊 Risultati e Valutazione
Performance Classificazione (media su 10 run)
```yaml
Patologia	      Modello	         Accuracy	        Precision	    Recall	      F1
Depressione	    Random Forest	   0.75 ± 0.03	    0.76 ± 0.04	  0.74 ± 0.03	  0.75
Diabete	        Random Forest	   0.88 ± 0.02	    0.89 ± 0.02	  0.87 ± 0.02	  0.88
Ipertensione	  Random Forest	   0.83 ± 0.03	    0.84 ± 0.03	  0.82 ± 0.03	  0.83
Media generale: 0.82 ± 0.05
```
Catene di Markov – Distribuzioni Stazionarie
```yaml
Patologia	        Low	    Medium	  High	  Stato Dominante
Depressione	        0.34	  0.40	    0.26	  Medium
Diabete	            0.24	  0.45	    0.31	  Medium
Ipertensione	    0.29	  0.38	    0.33	  Medium
```
## ⚠️ Disclaimer
### ATTENZIONE: Questo è un progetto accademico, non un dispositivo medico. 

COSA NON È:

❌ Non è un sistema diagnostico

❌ Non sostituisce la visita medica

❌ Non fornisce consigli medici validi

❌ Non è validato clinicamente

COSA È:

✅ Dimostrazione tecnologica di Ingegneria della Conoscenza

✅ Progetto didattico per esame universitario

✅ Esempio integrazione multi-modello AI

✅ Software open-source per scopi educativi

L'autore non è responsabile per eventuali usi impropri del software. Consultare sempre professionisti medici qualificati per questioni di salute.

## 📄 Licenza
MIT License

Copyright (c) 2026 Patruno Mirko

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

