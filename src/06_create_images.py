"""
Crea visualizzazioni essenziali per la relazione
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_PATH = PROJECT_ROOT / "docs" / "immagini"
DOCS_PATH.mkdir(parents=True, exist_ok=True)

def create_architecture_diagram():
    """Diagramma architettura sistema"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    components = [
        ('Knowledge Base\n(Prolog)', 0.2, 0.8, '#3498db'),
        ('Catene di Markov\n(Cap. 9)', 0.2, 0.5, '#e74c3c'),
        ('ML Classificazione\n(Cap. 7)', 0.5, 0.5, '#2ecc71'),
        ('CSP Ottimizzazione\n(Cap. 4)', 0.8, 0.5, '#f39c12'),
        ('Sistema Integrato', 0.5, 0.2, '#9b59b6')
    ]
    
    for name, x, y, color in components:
        circle = plt.Circle((x, y), 0.08, color=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', 
               fontsize=9, fontweight='bold', color='white')
    
    connections = [
        (0.2, 0.72, 0.2, 0.58),
        (0.2, 0.42, 0.5, 0.5),
        (0.5, 0.42, 0.8, 0.5),
        (0.8, 0.42, 0.5, 0.28),
    ]
    
    for x1, y1, x2, y2 in connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.set_title('Architettura Sistema Hygiea', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(DOCS_PATH / 'architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Creazione visualizzazioni...")
    
    # Crea tutte le immagini
    create_architecture_diagram()
    create_comprehensive_performance_plot()
    
    print(f"✓ Immagini salvate in: {DOCS_PATH}")
    
def create_comprehensive_performance_plot():
    """Crea grafico completo delle performance di tutti i modelli"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    DATA_PATH = Path(__file__).parent / "data"
    DOCS_PATH = Path(__file__).parent.parent / "docs" / "immagini"
    
    # Carica risultati
    try:
        df_results = pd.read_csv(DATA_PATH / "classification_results.csv")
    except:
        print("File risultati non trovato. Esegui prima la classificazione.")
        return
    
    # Dati di esempio se il file non esiste (per testing)
    if df_results.empty:
        data = {
            'Patologia': ['depression']*3 + ['diabetes']*3 + ['hypertension']*3,
            'Modello': ['Random Forest', 'SVM', 'Decision Tree']*3,
            'cv_mean': [0.75, 0.68, 0.72, 0.88, 0.82, 0.85, 0.83, 0.78, 0.80],
            'precision_mean': [0.76, 0.69, 0.73, 0.89, 0.83, 0.86, 0.84, 0.79, 0.81],
            'recall_mean': [0.74, 0.67, 0.71, 0.87, 0.81, 0.84, 0.82, 0.77, 0.79],
            'f1_mean': [0.75, 0.68, 0.72, 0.88, 0.82, 0.85, 0.83, 0.78, 0.80]
        }
        df_results = pd.DataFrame(data)
    
    # Crea grafico
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Grafico principale: barre raggruppate per patologia
    ax1 = plt.subplot(2, 2, (1, 2))
    
    models = df_results['Modello'].unique()
    diseases = df_results['Patologia'].unique()
    metrics = ['cv_mean', 'precision_mean', 'recall_mean', 'f1_mean']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Colori
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    disease_colors = {'depression': '#9b59b6', 'diabetes': '#1abc9c', 'hypertension': '#e67e22'}
    
    x = np.arange(len(metrics))
    width = 0.25
    
    # Per ogni modello, creo le barre
    for i, model in enumerate(models):
        model_data = df_results[df_results['Modello'] == model]
        values = []
        for metric in metrics:
            values.append(model_data[metric].mean())
        
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, values, width, label=model, 
                      color=colors[i], alpha=0.8, edgecolor='black')
        
        # Aggiungi valori
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Metriche')
    ax1.set_ylabel('Valore Medio')
    ax1.set_title('CONFRONTO GLOBALE DELLE METRICHE', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.legend(title='Modelli')
    
    # 2. Heatmap Accuracy per patologia
    ax2 = plt.subplot(2, 2, 3)
    
    # Crea matrice per heatmap
    heatmap_data = pd.pivot_table(df_results, 
                                  values='cv_mean', 
                                  index='Modello', 
                                  columns='Patologia')
    
    im = ax2.imshow(heatmap_data.values, cmap='RdYlGn', vmin=0.6, vmax=0.9)
    
    # Aggiungi testo nelle celle
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            text = ax2.text(j, i, f'{heatmap_data.values[i, j]:.3f}',
                           ha='center', va='center', 
                           color='white' if heatmap_data.values[i, j] < 0.75 else 'black')
    
    ax2.set_xticks(np.arange(len(heatmap_data.columns)))
    ax2.set_yticks(np.arange(len(heatmap_data.index)))
    ax2.set_xticklabels([d.title() for d in heatmap_data.columns])
    ax2.set_yticklabels(heatmap_data.index)
    ax2.set_title('Accuracy per Modello e Patologia', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Accuracy')
    
    # 3. Radar chart per Random Forest (miglior modello)
    ax3 = plt.subplot(2, 2, 4, polar=True)
    
    rf_data = df_results[df_results['Modello'] == 'Random Forest']
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for disease in diseases:
        values = []
        for metric in metrics:
            val = rf_data[rf_data['Patologia'] == disease][metric].values[0]
            values.append(val)
        values += values[:1]
        
        ax3.plot(angles, values, 'o-', linewidth=2, 
                label=disease.title(), color=disease_colors[disease])
        ax3.fill(angles, values, alpha=0.1, color=disease_colors[disease])
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metric_labels)
    ax3.set_ylim(0, 1)
    ax3.set_title('Random Forest - Performance per Patologia', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.suptitle('ANALISI COMPLETA DELLE PERFORMANCE DEI MODELLI DI CLASSIFICAZIONE\n' +
                '(Medie su 10 run, metriche macro)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Salva
    output_path = DOCS_PATH / "comprehensive_performance_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Analisi performance completa salvata: {output_path}")
    return output_path

if __name__ == "__main__":
    main()