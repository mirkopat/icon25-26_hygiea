"""
Constraint Satisfaction Problem (CSP) per raccomandazioni
Cap. 4 - Ragionamento con vincoli
Implementa: Arc Consistency + Backtracking
"""
import json
from pathlib import Path
from copy import deepcopy

DATA_PATH = Path(__file__).parent / "data"

class CSPSolver:
    """CSP con Arc Consistency e Backtracking"""
    
    def __init__(self, patient_risks):
        self.patient_risks = patient_risks
        
        # VARIABILI: slot di interventi da assegnare
        self.variables = ['slot_1', 'slot_2']
        
        # DOMINI: interventi disponibili + opzione 'none'
        self.interventions = {
            'exercise': {'cost': 2, 'eff': {'depression': 0.6, 'diabetes': 0.8, 'hypertension': 0.7}},
            'diet': {'cost': 1, 'eff': {'depression': 0.4, 'diabetes': 0.9, 'hypertension': 0.8}},
            'mindfulness': {'cost': 1, 'eff': {'depression': 0.8, 'hypertension': 0.6, 'diabetes': 0.3}},
            'sleep': {'cost': 1, 'eff': {'depression': 0.7, 'hypertension': 0.5, 'diabetes': 0.4}}
        }
        
        self.initial_domains = {
            'slot_1': list(self.interventions.keys()) + ['none'],
            'slot_2': list(self.interventions.keys()) + ['none']
        }
        
        # VINCOLI
        self.max_cost = 3
    
    def is_consistent(self, assignment):
        """Verifica consistenza vincoli"""
        # Vincolo 1: Costo totale ≤ 3
        total_cost = sum(
            self.interventions[v]['cost'] 
            for v in assignment.values() 
            if v != 'none'
        )
        if total_cost > self.max_cost:
            return False
        
        # Vincolo 2: No duplicati
        values = [v for v in assignment.values() if v != 'none']
        if len(values) != len(set(values)):
            return False
        
        # Vincolo 3: Almeno uno slot deve essere usato
        if all(v == 'none' for v in assignment.values()):
            return False
        
        return True
    
    def arc_consistency(self, domains):
        """Arc Consistency 3 (AC-3) semplificato"""
        changed = True
        
        while changed:
            changed = False
            
            for var1 in self.variables:
                for var2 in self.variables:
                    if var1 == var2:
                        continue
                    
                    # rimuove valori inconsistenti
                    to_remove = []
                    for val1 in domains[var1]:
                        # Verifica se esiste un valore in var2 consistente
                        has_support = False
                        for val2 in domains[var2]:
                            test_assignment = {var1: val1, var2: val2}
                            if self.is_consistent(test_assignment):
                                has_support = True
                                break
                        
                        if not has_support:
                            to_remove.append(val1)
                    
                    if to_remove:
                        for val in to_remove:
                            domains[var1].remove(val)
                        changed = True
        
        return domains
    
    def backtracking_search(self, assignment, domains):
        """Backtracking con pruning"""
        # Caso base: assignment completo
        if len(assignment) == len(self.variables):
            if self.is_consistent(assignment):
                return assignment
            return None
        
        # Scegle variabile non assegnata (MRV heuristic)
        unassigned = [v for v in self.variables if v not in assignment]
        var = min(unassigned, key=lambda v: len(domains[v]))
        
        # Prova ogni valore nel dominio
        for value in domains[var]:
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            # Verifica consistenza parziale
            if self.is_consistent(new_assignment):
                # Forward checking
                new_domains = deepcopy(domains)
                
                # Ricorsione
                result = self.backtracking_search(new_assignment, new_domains)
                if result is not None:
                    return result
        
        return None
    
    def evaluate_solution(self, assignment):
        """Valuta qualità soluzione"""
        interventions = [v for v in assignment.values() if v != 'none']
        
        if not interventions:
            return 0, {}
        
        total_cost = sum(self.interventions[i]['cost'] for i in interventions)
        
        # Calcola copertura patologie ad alto rischio
        coverage_score = 0
        high_risk_diseases = [d for d, r in self.patient_risks.items() if r == 'High']
        
        for disease in high_risk_diseases:
            disease_coverage = sum(
                self.interventions[i]['eff'].get(disease, 0)
                for i in interventions
            )
            coverage_score += min(disease_coverage, 1.0)
        
        # Penalità costo
        cost_penalty = (total_cost / self.max_cost) * 0.3
        
        final_score = coverage_score - cost_penalty
        
        return final_score, {
            'interventions': interventions,
            'cost': total_cost,
            'coverage': coverage_score,
            'score': final_score
        }
    
    def solve(self):
        """Risolvi CSP con AC-3 + Backtracking"""
        print("\n" + "="*60)
        print("CSP SOLVER - Arc Consistency + Backtracking")
        print("="*60)
        
        # 1. Arc Consistency
        print("\n1. Applicazione Arc Consistency...")
        domains = deepcopy(self.initial_domains)
        domains = self.arc_consistency(domains)
        
        print(f"Domini ridotti:")
        for var, domain in domains.items():
            print(f"  {var}: {domain}")
        
        # 2. Backtracking Search
        print("\n2. Backtracking Search...")
        solutions = []
        
        # Trova tutte le soluzioni
        for val1 in domains['slot_1']:
            for val2 in domains['slot_2']:
                assignment = {'slot_1': val1, 'slot_2': val2}
                if self.is_consistent(assignment):
                    score, details = self.evaluate_solution(assignment)
                    solutions.append((score, assignment, details))
        
        if not solutions:
            print("Nessuna soluzione trovata!")
            return None, None
        
        # Seleziona miglior soluzione
        solutions.sort(reverse=True, key=lambda x: x[0])
        best_score, best_assignment, best_details = solutions[0]
        
        print(f"\nSoluzioni trovate: {len(solutions)}")
        print(f"Miglior soluzione: {best_assignment}")
        
        return best_assignment, best_details

def main():
    # Profilo paziente esempio
    patient_risks = {
        'depression': 'High',
        'diabetes': 'High',
        'hypertension': 'Medium'
    }
    
    print("SISTEMA CSP - RACCOMANDAZIONI OTTIMALI")
    print("\nProfilo paziente:")
    for disease, risk in patient_risks.items():
        print(f"  {disease}: {risk}")
    
    # Risolvi CSP
    solver = CSPSolver(patient_risks)
    assignment, details = solver.solve()
    
    # Mostra risultati
    if details:
        print("\n" + "="*60)
        print("RACCOMANDAZIONI OTTIMALI")
        print("="*60)
        
        for intervention in details['interventions']:
            info = solver.interventions[intervention]
            print(f"\n• {intervention.title()}")
            print(f"  Costo: {info['cost']}/3")
            print(f"  Efficacia: {', '.join(f'{k}:{v:.1f}' for k,v in info['eff'].items() if v>0.5)}")
        
        print(f"\nCosto totale: {details['cost']}/3")
        print(f"Copertura: {details['coverage']:.2f}")
        print(f"Score: {details['score']:.2f}")
        
        # Salva
        output = {
            'patient_risks': patient_risks,
            'recommended': details['interventions'],
            'cost': details['cost'],
            'score': details['score']
        }
        
        with open(DATA_PATH / 'csp_solution.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print("\n✓ Soluzione salvata in 'data/csp_solution.json'")

if __name__ == "__main__":
    main()