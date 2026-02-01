"""
SISTEMA INTEGRATO HYGIEA - Versione semplice
Esegue tutto importando direttamente i moduli
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

def main():
    print("="*60)
    print("HYGIEA - Esecuzione sequenziale")
    print("="*60)
    
    # Esegue in sequenza chiamando le main() di ogni script
    scripts = [
        ("01_create_dataset", "Creazione dataset"),
        ("02_markov_model", "Analisi Markov"),
        ("03_classification", "Classificazione ML"),
        ("04_csp_recommender", "Raccomandazioni CSP"),
        ("05_integration", "Integrazione e report"),
        ("06_create_images", "Creazione immagini"),
        ("integration_demo", "Demo integrata")
    ]
    
    for mod_name, descrizione in scripts:
        print(f"\n▶ {descrizione}...")
        try:
            # Importa il modulo dinamicamente
            module = __import__(mod_name)
            # Esegue la funzione main() se esiste
            if hasattr(module, 'main'):
                module.main()
                print(f"✓ {descrizione} completato")
            else:
                print(f"⚠ {mod_name} non ha funzione main()")
        except ImportError as e:
            print(f"✗ Impossibile importare {mod_name}: {e}")
        except Exception as e:
            print(f"✗ Errore in {mod_name}: {e}")
    
    print("\n" + "="*60)
    print("SISTEMA COMPLETATO!")
    print("="*60)

if __name__ == "__main__":
    main()