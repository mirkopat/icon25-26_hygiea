"""
Catene di Markov con distribuzione stazionaria e simulazione
Cap. 9 - Ragionamento e Incertezza
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects  # per bordo al testo

DATA_PATH = Path(__file__).parent / "data"
DOCS_PATH = Path(__file__).parent.parent / "docs" / "immagini"
DOCS_PATH.mkdir(parents=True, exist_ok=True)

DISEASES = ['depression', 'diabetes', 'hypertension']


class MarkovChainAnalyzer:
    """Analisi Markov completa con distribuzione stazionaria"""
    
    def __init__(self):
        self.states = ['Low', 'Medium', 'High']
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
    
    def load_data(self):
        df = pd.read_csv(DATA_PATH / "simple_dataset.csv")
        print(f"✓ Dataset caricato: {len(df)} osservazioni")
        return df
    
    def calculate_transition_matrix(self, df, disease):
        """Calcola matrice di transizione 3×3"""
        P = np.zeros((3, 3))
        counts = np.zeros((3, 3))
        
        for patient_id in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == patient_id].sort_values('day')
            
            if len(patient_data) >= 2:
                risks = patient_data[f'{disease}_risk'].values
                
                for i in range(len(risks)-1):
                    from_state = self.state_to_idx[risks[i]]
                    to_state = self.state_to_idx[risks[i+1]]
                    counts[from_state, to_state] += 1
        
        # Normalizza per righe
        for i in range(3):
            row_sum = counts[i, :].sum()
            if row_sum > 0:
                P[i, :] = counts[i, :] / row_sum
            else:
                P[i, :] = 1/3
        
        return P
    
    def stationary_distribution(self, P, tol=1e-6, max_iter=1000):
        """Calcola distribuzione stazionaria π tale che πP = π"""
        pi = np.ones(3) / 3  # Inizializzazione uniforme
        
        for _ in range(max_iter):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi) < tol:
                return pi_new
            pi = pi_new
        
        return pi
    
    def mixing_time(self, P, epsilon=0.01):
        """Stima tempo di mixing (convergenza)"""
        pi_stationary = self.stationary_distribution(P)
        pi = np.ones(3) / 3
        
        for t in range(1, 100):
            pi = pi @ P
            total_variation = 0.5 * np.sum(np.abs(pi - pi_stationary))
            if total_variation < epsilon:
                return t
        
        return None
    
    def simulate_patient(self, P, initial_state, days=30):
        """Simula evoluzione paziente per N giorni"""
        trajectory = [initial_state]
        current_idx = self.state_to_idx[initial_state]
        
        for _ in range(days - 1):
            # Campiona prossimo stato dalla distribuzione P[current_idx]
            next_idx = np.random.choice(3, p=P[current_idx, :])
            trajectory.append(self.states[next_idx])
            current_idx = next_idx
        
        return trajectory
    
    def analyze_all_diseases(self):
        """Analisi completa per tutte le patologie"""
        df = self.load_data()
        print("="*60)
        print("ANALISI CATENE DI MARKOV COMPLETA")
        print("="*60)
        
        results = {}
        
        for disease in DISEASES:
            print(f"\n{'='*50}")
            print(f"{disease.upper()}")
            print('='*50)
            
            P = self.calculate_transition_matrix(df, disease)
            
            # Matrice di transizione
            print("\nMatrice di transizione P:")
            print("       Low    Med   High")
            for i, row in enumerate(P):
                print(f"{self.states[i]:5} {row[0]:.3f}  {row[1]:.3f}  {row[2]:.3f}")
            
            # Distribuzione stazionaria
            pi = self.stationary_distribution(P)
            print(f"\nDistribuzione stazionaria π:")
            for i, prob in enumerate(pi):
                print(f"  {self.states[i]}: {prob:.3f}")
            
            dominant_state = self.states[np.argmax(pi)]
            print(f"Stato dominante a lungo termine: {dominant_state}")
            
            # Tempo di mixing
            t_mix = self.mixing_time(P)
            if t_mix:
                print(f"Tempo di mixing (ε=0.01): {t_mix} giorni")
            
            # Stabilità
            stability = np.trace(P) / 3
            print(f"Stabilità (persistenza): {stability:.3f}")
            
            results[disease] = {
                'P': P,
                'pi': pi,
                'stability': stability,
                't_mix': t_mix
            }
        
        return results
    
    def visualize_results(self, results):
        """Visualizza matrici e distribuzioni con layout ottimizzato"""
        # Crea figure con dimensioni maggiori e layout ottimizzato
        fig = plt.figure(figsize=(16, 10))
        
        # Mi definisce il layout della griglia
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
        
        # Riga 1: Matrici di transizione
        for idx, disease in enumerate(DISEASES):
            ax = fig.add_subplot(gs[0, idx])
            P = results[disease]['P']
            
            # Heatmap con annotazioni più leggibili
            im = ax.imshow(P, cmap='YlOrRd', vmin=0, vmax=1)
            
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(self.states, fontsize=11, fontweight='bold')
            ax.set_yticklabels(self.states, fontsize=11, fontweight='bold')
            
            ax.set_title(f'{disease.title()}\nMatrice di Transizione', 
                        fontsize=12, fontweight='bold', pad=15)
            
            for i in range(3):
                for j in range(3):
                    text_color = 'white' if P[i, j] > 0.5 else 'black'
                    text = ax.text(j, i, f'{P[i,j]:.2f}', 
                                  ha='center', va='center',
                                  color=text_color, fontsize=11,
                                  fontweight='bold' if P[i,j] > 0.7 else 'normal')
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=2, foreground='white' if text_color == 'black' else 'black'),
                        path_effects.Normal()
                    ])
            
            ax.set_xlabel('Stato Successivo', fontsize=10, labelpad=10)
            ax.set_ylabel('Stato Corrente', fontsize=10, labelpad=10)
        
        # Riga 2: Distribuzioni stazionarie
        for idx, disease in enumerate(DISEASES):
            ax = fig.add_subplot(gs[1, idx])
            pi = results[disease]['pi']
            
            # Barre con colori distintivi
            colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Verde, Arancione, Rosso
            bars = ax.bar(self.states, pi, color=colors, edgecolor='black', linewidth=1.5)
            
            ax.set_ylim(0, 1)
            ax.set_title(f'{disease.title()}\nDistribuzione Stazionaria', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.set_ylabel('Probabilità', fontsize=10, labelpad=10)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            # Aggiunge valori sopra le barre
            for i, (bar, prob) in enumerate(zip(bars, pi)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{prob:.3f}', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
            
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            
            # Questo mi evidenzia lo stato dominante
            dominant_idx = np.argmax(pi)
            bars[dominant_idx].set_alpha(0.9)
            bars[dominant_idx].set_linewidth(2.5)
            bars[dominant_idx].set_edgecolor('darkblue')
            
            # Annoto stato dominante
            ax.annotate(f'DOMINANTE: {self.states[dominant_idx]}', 
                       xy=(dominant_idx, pi[dominant_idx]), 
                       xytext=(dominant_idx, pi[dominant_idx] + 0.1),
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))
        
        cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar_ax.set_ylabel('Probabilità di Transizione', fontsize=10, labelpad=10)
        
        plt.suptitle('ANALISI CATENE DI MARKOV - TRANSIZIONI RISCHIO PAZIENTE', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        info_text = (
            "Matrici di transizione 3×3 (stati: Low, Medium, High)\n"
            f"Dati: {len(self.load_data())} osservazioni | 60 pazienti × 7 giorni"
        )
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')
        
        output_path = DOCS_PATH / "markov_complete_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"\n✓ Grafico salvato: {output_path}")
        print(f"  Dimensioni: 1600×1000 pixels")
        print(f"  Risoluzione: 300 DPI")


def main():
    analyzer = MarkovChainAnalyzer()
    
    # Analisi completa
    results = analyzer.analyze_all_diseases()
    
    # Simulazione esempio
    print("\n" + "="*60)
    print("SIMULAZIONE EVOLUZIONE PAZIENTE (30 giorni)")
    print("="*60)
    
    np.random.seed(42)
    for disease in DISEASES:
        P = results[disease]['P']
        trajectory = analyzer.simulate_patient(P, 'Medium', days=30)
        
        print(f"\n{disease.title()} (start: Medium):")
        print(f"Giorni 1-10:  {' → '.join(trajectory[:10])}")
        print(f"Giorni 20-30: {' → '.join(trajectory[19:30])}")
        
        final_state = trajectory[-1]
        print(f"Stato finale (giorno 30): {final_state}")
    
    # Visualizzazione
    analyzer.visualize_results(results)
    
    print("\n" + "="*60)
    print("CONCLUSIONI MARKOV")
    print("="*60)
    print("1. Distribuzione stazionaria mostra rischio a lungo termine")
    print("2. Tempo di mixing indica rapidità di convergenza")
    print("3. Simulazioni utili per previsioni individuali")


if __name__ == "__main__":
    main()