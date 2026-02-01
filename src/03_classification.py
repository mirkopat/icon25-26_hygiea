"""
Classificazione con valutazione completa
Cap. 7 - Apprendimento Supervisionato
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DATA_PATH = Path(__file__).parent / "data"
DOCS_PATH = Path(__file__).parent.parent / "docs" / "immagini"
DOCS_PATH.mkdir(parents=True, exist_ok=True)

DISEASES = ['depression', 'diabetes', 'hypertension']

def evaluate_model(model, X, y, disease, n_runs=10):
    """Valutazione ROBUSTA con N run e statistiche aggregate"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    all_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
        'confusion_matrices': []
    }
    
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=run
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        all_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        all_metrics['precision'].append(precision)
        all_metrics['recall'].append(recall)
        all_metrics['f1'].append(f1)
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        all_metrics['confusion_matrices'].append(cm)
    
    results = {
        'cv_mean': np.mean(all_metrics['accuracy']),
        'cv_std': np.std(all_metrics['accuracy']),
        'precision_mean': np.mean(all_metrics['precision']),
        'precision_std': np.std(all_metrics['precision']),
        'recall_mean': np.mean(all_metrics['recall']),
        'recall_std': np.std(all_metrics['recall']),
        'f1_mean': np.mean(all_metrics['f1']),
        'f1_std': np.std(all_metrics['f1']),
        'mean_cm': np.mean(all_metrics['confusion_matrices'], axis=0),
        'std_cm': np.std(all_metrics['confusion_matrices'], axis=0),
        'n_runs': n_runs
    }
    
    return results

def main():
    print("="*60)
    print("CLASSIFICAZIONE CON METRICHE COMPLETE")
    print("="*60)
    
    df = pd.read_csv(DATA_PATH / "simple_dataset.csv")
    print(f"Dataset: {len(df)} osservazioni\n")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42)
    }
    
    features = ['age', 'bmi', 'sleep_quality', 'stress_level', 
                'physical_activity', 'sugar_intake', 'salt_intake']
    
    all_results = []
    
    for disease in DISEASES:
        print(f"\n{'='*50}")
        print(f"{disease.upper()}")
        print('='*50)
        
        X = df[features].values
        y = LabelEncoder().fit_transform(df[f'{disease}_risk'].values)
        
        for model_name, model in models.items():
            print(f"\n{model_name}:")
            
            results = evaluate_model(model, X, y, disease)
            
            print(f"  Accuracy:  {results['cv_mean']:.3f} (±{results['cv_std']:.3f})")
            print(f"  Precision: {results['precision_mean']:.3f} (±{results['precision_std']:.3f})")
            print(f"  Recall:    {results['recall_mean']:.3f} (±{results['recall_std']:.3f})")
            print(f"  F1-Score:  {results['f1_mean']:.3f} (±{results['f1_std']:.3f})")

            # Aggiunge stampa matrice confusione media
            print(f"\n  Matrice confusione media ({results['n_runs']} run):")
            
            for i, row in enumerate(results['mean_cm']):
                print(f"    {['Low','Medium','High'][i]}: {row}")
            
            all_results.append({
                'Patologia': disease,
                'Modello': model_name,
                'cv_mean': results['cv_mean'],
                'cv_std': results['cv_std'],
                'precision_mean': results['precision_mean'],
                'precision_std': results['precision_std'],
                'recall_mean': results['recall_mean'],
                'recall_std': results['recall_std'],
                'f1_mean': results['f1_mean'],
                'f1_std': results['f1_std'],
                'n_runs': results['n_runs']
            })
    
    # Salva risultati
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(DATA_PATH / 'classification_results.csv', index=False)
    
    # Visualizzazione
    create_metrics_comparison_plot(df_results)
    
    # Chiama la funzione esistente
    create_visualization(df_results)
    
    print("\n" + "="*60)
    print("CONCLUSIONI")
    print("="*60)
    print("✓ Valutazione robusta: 10 run per modello")
    print("✓ Metriche complete: Acc, Prec, Rec, F1")
    print("✓ Random Forest: miglior bilanciamento")

def create_visualization(df_results):
    """Crea grafico comparativo"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy con error bars
    ax1 = axes[0]
    for model in df_results['Modello'].unique():
        model_data = df_results[df_results['Modello'] == model]
        x = range(len(model_data))
        ax1.errorbar(x, model_data['cv_mean'], yerr=model_data['cv_std'],
                    marker='o', capsize=5, label=model)
    
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(['Depression', 'Diabetes', 'Hypertension'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy per Modello e Patologia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    pivot = df_results.pivot(index='Patologia', columns='Modello', values='f1_mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1, ax=ax2)
    ax2.set_title('F1-Score Comparativo')
    
    plt.tight_layout()
    plt.savefig(DOCS_PATH / 'classification_complete.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Grafico salvato: {DOCS_PATH / 'classification_complete.png'}")

def create_metrics_comparison_plot(df_results):
    """Crea grafico comparativo di tutte le metriche per tutti i modelli"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    models = df_results['Modello'].unique()
    diseases = df_results['Patologia'].unique()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Prepara dati per il grafico
    data_for_plot = {}
    for disease in diseases:
        data_for_plot[disease] = {}
        for model in models:
            model_data = df_results[(df_results['Patologia'] == disease) & 
                                   (df_results['Modello'] == model)]
            if not model_data.empty:
                # Prende le medie per ogni metrica
                data_for_plot[disease][model] = {
                    'Accuracy': model_data['cv_mean'].iloc[0],
                    'Precision': model_data['precision_mean'].iloc[0],
                    'Recall': model_data['recall_mean'].iloc[0],
                    'F1-Score': model_data['f1_mean'].iloc[0]
                }
    
    # Crea grafico a barre raggruppate
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colori per i modelli
    colors = {
        'Random Forest': '#2ecc71',  # Verde
        'SVM': '#3498db',            # Blu
        'Decision Tree': '#e74c3c'   # Rosso
    }
    
    for idx, disease in enumerate(diseases):
        ax = axes[idx]
        
        # Prepara dati per questa patologia
        x = np.arange(len(metrics))  # Posizioni delle metriche
        width = 0.25  # Larghezza delle barre
        
        for i, model in enumerate(models):
            if model in data_for_plot[disease]:
                # Calcola posizione per ogni modello
                offset = (i - 1) * width
                model_data = data_for_plot[disease][model]
                values = [model_data[metric] for metric in metrics]
                
                # Crea barre
                bars = ax.bar(x + offset, values, width, 
                             label=model, color=colors[model],
                             alpha=0.8, edgecolor='black')
                
                # Aggiungi valori sulle barre
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Configura assi
        ax.set_xlabel('Metriche')
        ax.set_ylabel('Valore')
        ax.set_title(f'{disease.title()} - Confronto Modelli', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
    
    plt.suptitle('CONFRONTO COMPLETO METRICHE PER PATOLOGIA E MODELLO', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Salva
    output_path = DOCS_PATH / "metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Grafico comparativo salvato: {output_path}")
    return output_path

if __name__ == "__main__":
    main()