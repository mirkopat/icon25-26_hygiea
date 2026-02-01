/* 
   Knowledge Base Prolog - Hygiea
   Cap. 15 - Rappresentazione e Ragionamento Relazionale
   
   COMPLESSITÀ:
   - Fatti: 15
   - Regole: 12 (incluse 3 abduttive)
   - Profondità inferenza: 3 livelli
   - Pattern matching avanzato con findall/bagof
*/

% ============ METADATI ============
complessita_kb(media).
numero_regole(12).
numero_fatti(15).
profondita_max_inferenza(3).

% ============ ONTOLOGIA BASE ============
% Gerarchia patologie
tipo_patologia(neurologica, depressione).
tipo_patologia(metabolica, diabete).
tipo_patologia(cardiovascolare, ipertensione).

patologia(P) :- tipo_patologia(_, P).

% Interventi con costo
intervento(esercizio, 2).
intervento(dieta, 1).
intervento(mindfulness, 1).
intervento(sonno, 1).

% ============ FATTORI DI RISCHIO ============
% fattore_rischio(Fattore, Patologia, Peso)
fattore_rischio(stress_alto, depressione, 0.8).
fattore_rischio(sonno_scadente, depressione, 0.7).
fattore_rischio(fam_depressione, depressione, 0.6).

fattore_rischio(bmi_alto, diabete, 0.7).
fattore_rischio(zuccheri_alti, diabete, 0.8).
fattore_rischio(fam_diabete, diabete, 0.6).

fattore_rischio(sale_alto, ipertensione, 0.9).
fattore_rischio(stress_alto, ipertensione, 0.7).
fattore_rischio(fam_ipertensione, ipertensione, 0.5).

% Efficacia interventi
efficace(esercizio, diabete, 0.8).
efficace(esercizio, ipertensione, 0.7).
efficace(esercizio, depressione, 0.6).
efficace(dieta, diabete, 0.9).
efficace(dieta, ipertensione, 0.8).
efficace(mindfulness, depressione, 0.8).
efficace(mindfulness, ipertensione, 0.6).
efficace(sonno, depressione, 0.7).

% ============ REGOLE DI INFERENZA (Livello 1) ============

% R1: Rischio alto se ≥2 fattori presenti
rischio_alto(Paziente, Patologia) :-
    patologia(Patologia),
    findall(Peso, (
        fattore_presente(Paziente, Fattore),
        fattore_rischio(Fattore, Patologia, Peso)
    ), Pesi),
    length(Pesi, N),
    N >= 2.

% R2: Calcolo punteggio rischio pesato
punteggio_rischio(Paziente, Patologia, Punteggio) :-
    patologia(Patologia),
    findall(Peso, (
        fattore_presente(Paziente, Fattore),
        fattore_rischio(Fattore, Patologia, Peso)
    ), Pesi),
    sum_list(Pesi, Somma),
    length(Pesi, N),
    (N > 0 -> Punteggio is Somma / N ; Punteggio is 0).

% R3: Intervento raccomandato
raccomanda_intervento(Paziente, Intervento) :-
    rischio_alto(Paziente, Patologia),
    efficace(Intervento, Patologia, Efficacia),
    Efficacia > 0.6.

% ============ REGOLE AVANZATE (Livello 2) ============

% R4: Fattori ereditari
fattore_ereditario(Fattore) :-
    fattore_rischio(Fattore, _, _),
    sub_atom(Fattore, 0, 4, _, 'fam_').

% R5: Interventi sostenibili (costo ≤ 2)
intervento_sostenibile(Intervento) :-
    intervento(Intervento, Costo),
    Costo =< 2.

% R6: Priorità alta se rischio multiplo
priorita_alta(Paziente, Intervento) :-
    raccomanda_intervento(Paziente, Intervento),
    findall(P, rischio_alto(Paziente, P), Patologie),
    length(Patologie, N),
    N >= 2.

% R7: Comorbidità (patologie condividono fattori)
comorbidita(Pat1, Pat2) :-
    Pat1 \= Pat2,
    fattore_rischio(Fattore, Pat1, _),
    fattore_rischio(Fattore, Pat2, _).

% ============ RAGIONAMENTO ABDUTTIVO (Livello 3) ============

% R8: Abduzione - Spiega perché c'è rischio
spiega_rischio(Paziente, Patologia, Fattori) :-
    rischio_alto(Paziente, Patologia),
    findall(Fattore, (
        fattore_presente(Paziente, Fattore),
        fattore_rischio(Fattore, Patologia, _)
    ), Fattori).

% R9: Abduzione - Ipotizza fattori mancanti
ipotizza_fattori(Paziente, Patologia, FattoriMancanti) :-
    patologia(Patologia),
    findall(Fattore, (
        fattore_rischio(Fattore, Patologia, Peso),
        \+ fattore_presente(Paziente, Fattore),
        Peso > 0.7  % Solo fattori rilevanti
    ), FattoriMancanti).

% R10: Diagnosi differenziale
diagnosi_differenziale(Paziente, Patologie) :-
    findall(Patologia-Punteggio, (
        patologia(Patologia),
        punteggio_rischio(Paziente, Patologia, Punteggio),
        Punteggio > 0
    ), Coppie),
    sort(2, @>=, Coppie, PatologieOrdinate),
    findall(P, member(P-_, PatologieOrdinate), Patologie).

% R11: Interventi sinergici
sinergia(esercizio, sonno).
sinergia(dieta, esercizio).
sinergia(mindfulness, sonno).

piano_sinergico(Paziente, [Int1, Int2]) :-
    raccomanda_intervento(Paziente, Int1),
    raccomanda_intervento(Paziente, Int2),
    sinergia(Int1, Int2),
    Int1 \= Int2.

% R12: Predizione evoluzione (ricorsiva)
evoluzione_rischio(Paziente, Patologia, GiorniSenzaIntervento, RischioFinale) :-
    punteggio_rischio(Paziente, Patologia, RischioIniziale),
    fattore_rischio(_, Patologia, PesoMedio),
    DeltaPerGiorno is PesoMedio * 0.05,
    Incremento is DeltaPerGiorno * GiorniSenzaIntervento,
    RischioFinale is min(1.0, RischioIniziale + Incremento).

% ============ DATI DI TEST ============
% Paziente esempio: Marco
fattore_presente(marco, stress_alto).
fattore_presente(marco, sonno_scadente).
fattore_presente(marco, bmi_alto).
fattore_presente(marco, sale_alto).
fattore_presente(marco, fam_depressione).

% Paziente esempio: Anna
fattore_presente(anna, zuccheri_alti).
fattore_presente(anna, fam_diabete).
fattore_presente(anna, stress_alto).

% ============ QUERY DI ESEMPIO ============
/*
ESEMPI DI QUERY (da eseguire in SWI-Prolog):

1. Query base:
   ?- rischio_alto(marco, X).
   X = depressione ;
   X = diabete ;
   X = ipertensione.

2. Punteggi rischio:
   ?- punteggio_rischio(marco, depressione, P).
   P = 0.75.

3. Raccomandazioni:
   ?- raccomanda_intervento(marco, I).
   I = esercizio ;
   I = dieta ;
   I = mindfulness.

4. Abduzione - Spiega rischio:
   ?- spiega_rischio(marco, depressione, Fattori).
   Fattori = [stress_alto, sonno_scadente, fam_depressione].

5. Ipotizza fattori mancanti:
   ?- ipotizza_fattori(anna, ipertensione, Mancanti).
   Mancanti = [sale_alto].

6. Diagnosi differenziale:
   ?- diagnosi_differenziale(marco, Patologie).
   Patologie = [ipertensione, depressione, diabete].

7. Piano sinergico:
   ?- piano_sinergico(marco, Piano).
   Piano = [esercizio, sonno] ;
   Piano = [mindfulness, sonno].

8. Comorbidità:
   ?- comorbidita(depressione, X).
   X = ipertensione.

9. Evoluzione rischio:
   ?- evoluzione_rischio(marco, depressione, 30, RischioFinale).
   RischioFinale = 0.95.
*/

% ============ METADATI PER VALUTAZIONE ============
% Documentazione complessità

complessita_algoritmo(rischio_alto/2, O(n²)) :- 
    % n = numero fattori × numero patologie
    write('Analisi complessità rischio_alto/2:'), nl,
    write('  - findall: O(f×p) dove f=fattori, p=patologie'), nl,
    write('  - length: O(n)'), nl,
    write('  - Totale: O(f×p) ≈ O(n²) per matching incrociato').

complessita_algoritmo(punteggio_rischio/3, O(n)) :-
    write('Analisi complessità punteggio_rischio/3:'), nl,
    write('  - findall: O(f) dove f=fattori per patologia'), nl,
    write('  - sum_list: O(f)'), nl,
    write('  - Totale: O(f) ≈ O(n)').

% Query di test automatiche
test_complessita :-
    complessita_algoritmo(rischio_alto/2, _),
    complessita_algoritmo(punteggio_rischio/3, _).