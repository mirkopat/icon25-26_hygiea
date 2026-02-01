"""
Dataset semplificato per 3 patologie
60 pazienti × 7 giorni = 420 osservazioni
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configurazione
np.random.seed(42)
DATA_PATH = Path(__file__).parent / "data"
DATA_PATH.mkdir(exist_ok=True)

DISEASES = ['depression', 'diabetes', 'hypertension']

def generate_simple_dataset(n_patients=60, n_days=7):
    """Genera dataset semplice per 3 patologie"""
    print("Generazione dataset semplificato...")
    
    data = []
    
    for patient_id in range(n_patients):
        age = np.random.randint(20, 80)
        gender = np.random.choice(['M', 'F'])
        bmi = np.random.uniform(18, 35)
        
        family_history = {
            'depression': np.random.random() < 0.2,
            'diabetes': np.random.random() < 0.25,
            'hypertension': np.random.random() < 0.3
        }
        
        for day in range(1, n_days + 1):
            sleep_quality = np.random.randint(1, 10)
            stress_level = np.random.randint(1, 10)
            physical_activity = np.random.randint(0, 120)
            sugar_intake = np.random.randint(10, 100)
            salt_intake = np.random.randint(2, 15)
            
            depression_risk = (
                (stress_level / 10 * 0.4) +
                ((10 - sleep_quality) / 9 * 0.3) +
                (family_history['depression'] * 0.3)
            ) + np.random.normal(0, 0.05)
            
            diabetes_risk = (
                (bmi - 25) / 20 * 0.4 +
                (sugar_intake / 100 * 0.3) +
                (1 - physical_activity / 120) * 0.3
            ) + np.random.normal(0, 0.05)
            
            hypertension_risk = (
                (salt_intake / 15 * 0.5) +
                (stress_level / 10 * 0.3) +
                ((age - 40) / 40 * 0.2 if age > 40 else 0)
            ) + np.random.normal(0, 0.05)
            
            def classify_risk(risk):
                risk = max(0, min(1, risk))
                return 'High' if risk > 0.6 else 'Medium' if risk > 0.3 else 'Low'
            
            record = {
                'patient_id': patient_id,
                'day': day,
                'age': age,
                'gender': gender,
                'bmi': round(bmi, 1),
                'sleep_quality': sleep_quality,
                'stress_level': stress_level,
                'physical_activity': physical_activity,
                'sugar_intake': sugar_intake,
                'salt_intake': salt_intake,
                'depression_risk_score': round(depression_risk, 3),
                'diabetes_risk_score': round(diabetes_risk, 3),
                'hypertension_risk_score': round(hypertension_risk, 3),
                'depression_risk': classify_risk(depression_risk),
                'diabetes_risk': classify_risk(diabetes_risk),
                'hypertension_risk': classify_risk(hypertension_risk)
            }
            
            data.append(record)
    
    df = pd.DataFrame(data)
    return df

def create_simple_kb():
    """Crea Knowledge Base semplice in JSON"""
    kb = {
        "diseases": DISEASES,
        "risk_factors": {
            "depression": ["stress_level", "sleep_quality", "family_history"],
            "diabetes": ["bmi", "sugar_intake", "physical_activity"],
            "hypertension": ["salt_intake", "stress_level", "age"]
        },
        "interventions": {
            "exercise": {
                "name": "Esercizio Fisico",
                "cost": 2,
                "targets": ["diabetes", "hypertension", "depression"],
                "effects": ["reduce_bmi", "reduce_stress", "improve_sleep"]
            },
            "diet": {
                "name": "Dieta Controllata",
                "cost": 1,
                "targets": ["diabetes", "hypertension"],
                "effects": ["reduce_sugar", "reduce_salt"]
            },
            "mindfulness": {
                "name": "Mindfulness",
                "cost": 1,
                "targets": ["depression", "hypertension"],
                "effects": ["reduce_stress", "improve_sleep"]
            }
        }
    }
    return kb

def main():
    print("="*60)
    print("HYGIEA - Dataset Semplificato")
    print("="*60)
    
    df = generate_simple_dataset()
    df.to_csv(DATA_PATH / "simple_dataset.csv", index=False)
    
    print(f"\n✓ Dataset creato: {len(df)} osservazioni")
    print(f"✓ Pazienti: {df['patient_id'].nunique()}")
    print(f"✓ Giorni: {df['day'].max()}")
    print(f"✓ Patologie: {len(DISEASES)}")
    
    print("\nDistribuzione Rischio:")
    for disease in DISEASES:
        risk_counts = df[f'{disease}_risk'].value_counts(normalize=True)
        print(f"\n{disease.title()}:")
        for risk, perc in risk_counts.items():
            print(f"  {risk}: {perc:.1%}")
    
    kb = create_simple_kb()
    with open(DATA_PATH / "wellness_kb.json", 'w') as f:
        json.dump(kb, f, indent=2)
    
    print(f"\n✓ Knowledge Base salvata")

if __name__ == "__main__":
    main()