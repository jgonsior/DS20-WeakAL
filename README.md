# Weak Supervision Experiments


**Story:**  


Labeln von Daten macht keinen Spaß, daher:  


--> AL (keine redundanten Daten erneut labeln)  

--> weak supervision (viele Daten mit schlechten Label sind besser als wenige Daten mit guten Label)  


**Problem von Snorkel:**  

*   in Python geschriebene LF sind zu komplex für Domänenexperten außerhalb der Informatik und nicht einfach anwendbar für Probleme außerhalb der Textklassifikation (Tabellen etc.)  

**Problem von SNUBA:**  


*   zu starker Fokus im Konzept auf binäre Klassifikationsprobleme (Heuristiken werden immer nur für eine Klasse generiert -> kein diverses Ergebnis) -> Probleme von imbalanced datasets werden nicht behandelt  

*   die labels der automatisch generierte Heuristiken sind vielleicht doch zu "weak"?  

*   Domänenexperten werden nur noch für die Erstellung von ground truth data verwendet -> aber ihr Wissen über sinnvolle und nicht sinnvolle Heuristiken wird ignoriert!  

**Mögliche weiterführende Arbeiten:**  


*   Verwendung von schwierigen imbalanced datasets mit mehreren Labels wie Tabellenklassifikation für die LF zu komplex für Domänenexperten sind  

*   Untersuchung der vorgeschlagenen Parameter von SNUBA mit Hinblick auf konkreten Einfluss von bewusst falschen "weakly" labeled data (mit Bezug auf verschiedene Probleme von imbalanced datasets):  

*   keine weak labels für kleine Klassen, für diese nur Abstain  

*   weak labels die sich nur auf bestimmte features konzentrieren (Diversität von snuba)  

*   wo ist sweet spot zwischen accuracy und coverage (also optimale Menge an "halbwegs" korrekt gelabelten Daten und "halbwegs" vielen Daten? (bei SNUBA einfaches 1:1 Verhältnis von Accuracy und Jaccard)  

*   Untersuchung von Gewichten für "strong" (also manuell erstellten) und "weak" labels (Gewicht von weakly labeled Datenpunkt von minority klasse höher als von weakly labeled dominanter klasse, aber kleiner als Gewicht von strongly gelabelten Datenpunkt der dominanten klasse?!)  

*   Unterschung wie "tief" die shallow Decision Trees (oder andere Verfahren) für die Heuristiken sein sollen bevor sie zu überangepasst werden  

*   welche Eigenschaften muss der kleine gelabelte Datensatz mit sich bringen um sinnvoll als Datenbasis für automatisch generierte Heuristiken zu verwenden ist (unter Berücksichtigung des großen ungelabelten Datensatzes -> sind alle vorhandenen Cluster in ihm mit labeln des gelabelten Datensatzes versehen)?  

*   Erweiterung von AL cycle durch vorgeschlagene Heuristiken ("basierend auf den bisherigen Labels, sollen diese 30 ähnlichen Datenpunkte auch so gelabeld werden?")  

*   Problemstellung: wann kann eine "sinnvolle" Heuristik vorgeschlagen werden? verwendete Parameter mit vorherigen Ergebnissen begründen  

*   Visualisierung der verwendeten Heuristik für Benutzer -> Möglichkeit diese noch zu ändern, denn für Domänenexperten ist deutlich einfacher eine bereits existierende formulierte Heuristik zu verändern bzw. anzupassen als selber von scratch zu arbeiten (Anpassung von Parametern in DTree, oder hinzunahme von anderen Datenpunkten um Parameter erneut anzupassen, etc.)  

--> **Reimplementierung als "Snuba Lite":** im Gegensatz zum richtigen Snuba ist die Ausgabe für einen gegebenen Datensatz nur eine Heuristik (und nicht mehrere) und dazu eine Confidence-Maß wie sinnvoll diese Heuristik wohl ist -> Hauptproblem ist die Berechnung dieses Confidence Maßes mit Hinblick auf imblanced multiclass datasets  


**Unser Beitrag:**  


*   Berechnung des Confidence Maßes der Heuristiken mit Hinblick auf komplexe imbalanced multiclass Probleme (wann abstain, heuristiken pro klasse & pro feature, sweet spot accuracy & coverage, gewichtung von labels, wann sollte eine Heuristik vorgeschlagen werden, wie tief sollten die shallow decision trees sein, …)  

*   Integration von vorgeschlagenen, menschenverständlichen Heuristiken (z. B. visualisierte Decision Trees, Darstellung von Cluster, …), in AL cycle  

Parallel zur Entwicklung von SNUBA Lite erweitern Studierende AERGIA als proof-of-concept Implementierung mittels Gamification von AL mit den vorgeschlagenen LF  
