# Thesis Project - Semi-Supervised Learning


---

Executive Summary 

The thesis explores semi-supervised learning approaches in scenarios where labeled data are limited while unlabeled data are widely available. In recent years, the increasing availability of large-scale datasets has highlighted the limitations of traditional supervised learning methods, since obtaining reliable labels is often expensive, time-consuming, and impractical in real-world applications. Semi-supervised learning addresses this issue by integrating both labeled and unlabeled data during the learning process.

The main focus of the thesis is the analysis of semi-supervised classification methods, investigating how unlabeled samples can improve class assignment performance when only a small amount of labeled data is available. In addition, semi-supervised clustering techniques were studied to evaluate how partial supervision can provide preliminary information about cluster structure and data organization.

Several algorithms, including S3VM, Laplacian-based methods, MPCK-Means, and Semi-Supervised Spectral Clustering, were implemented and evaluated through experiments on multiple UCI datasets. Experimental results on classification tasks showed that increasing the amount of labeled data generally improves accuracy and stability, although performance remains strongly influenced by dataset characteristics such as class separability, overlap, and dimensionality. The analysis also highlighted that S3VM methods are often competitive with very limited supervision, while Laplacian-based approaches tend to achieve better performance as the number of labeled samples increases.

For semi-supervised clustering, the introduction of pairwise constraints generally improves clustering quality, with the largest performance gains often achieved using a relatively small number of constraints before reaching saturation effects. The experiments further showed that the stability and effectiveness of clustering methods are closely related to the intrinsic structure of the data and to the adopted constraint configuration.

---

🧩 Gameplay

In Obscura il giocatore esplora un ospedale psichiatrico abbandonato in prima persona, cercando di sopravvivere mentre ricostruisce un passato oscuro e frammentato.

Le principali meccaniche di gioco includono:

- Esplorazione immersiva 🕯️
  Ambienti labirintici e in continua trasformazione, con corridoi che cambiano, porte che si aprono solo in determinate condizioni e stanze che rivelano dettagli inquietanti.

- Enigmi psicologici 🧩
  Puzzle ambientali, codici nascosti nei documenti clinici e indovinelli che richiedono osservazione e logica per essere risolti.

- Gestione delle risorse 🔦
  Il giocatore deve utilizzare con cautela torce e batterie, trovando oggetti che svelano indizi ma che spesso richiedono scelte su come e quando impiegarli.

- Minacce dinamiche 👁️‍🗨️
  Presenze oscure che si muovono in modo imprevedibile e reagiscono al rumore e alle azioni del giocatore, creando tensione costante e costringendo a bilanciare stealth e fuga.

- Narrazione ambientale 📜
  Documenti, graffiti e oggetti raccontano storie frammentarie, rivelando progressivamente dettagli sul passato del luogo e sul ruolo del protagonista.

- Flashback inquietanti 🩸
  In momenti chiave il protagonista rivive frammenti disturbanti del suo passato. Queste sequenze alterano l’ambiente e la percezione del giocatore, offrendo nuove informazioni ma aumentando il senso di disagio e confusione.

- Interazione con il microfono 🎤
  Il gioco potrà utilizzare il microfono del giocatore per captare suoni reali. Ogni rumore emesso (parlare, respirare forte, piccoli colpi) potrà attirare l’attenzione delle creature, aumentando la tensione e obbligando il giocatore a restare davvero in silenzio nella vita reale.

---

🛠️ Tecnologie Utilizzate

Unreal Engine 5.0

---

📌 Stato del Progetto

✅ Funzionalità Completate

- Sistema delle porte → apertura e chiusura interattiva.
- Torcia → accensione/spegnimento, gestione della luce.
- Sistema dei passi → suoni diversi su erba, legno e acciaio.
- Nemico base → pattugliamento, inseguimento e attacco.
- Gestione luci → accensione, spegnimento, rottura e sfarfallio dinamico.
- Sistema di salute → danni e morte del personaggio.
- Stamina + Sprint → corsa limitata con barra stamina.
- Sistema batterie → gestione della durata e ricarica della torcia.
- Sistema battito cardiaco → suono che aumenta d’intensità e ritmo in base alla distanza del nemico.
- Sistema documenti → raccolta e lettura di file/testimonianze per svelare la storia.
  
🔄 In Sviluppo

 - Orologio con GUI → interfaccia per mostrare vita, stamina e livello di paura.
 - IA del mostro avanzata → rilevamento solo con luce accesa.
 - Sistema paura → meccanica psicologica che influenza la percezione del giocatore.


---
 
🔮 Visione futura

In futuro, Obscura potrà espandersi con nuove funzionalità e contenuti per arricchire l’esperienza di gioco:

- Flashback interattivi → sequenze narrative che immergono il giocatore nei ricordi inquietanti del protagonista.
- Puzzle complessi → enigmi ambientali più articolati, collegati a documenti e indizi nascosti.
- Finali multipli → scelte del giocatore che influenzano la trama e il destino del protagonista.
- Nuove aree dell’ospedale → reparti inesplorati, sotterranei e ambienti distorti che ampliano la mappa.
- Nemici aggiuntivi → creature con comportamenti unici che rappresentano diversi aspetti della follia e del senso di colpa.
- Sistema audio avanzato → utilizzo dinamico dei suoni per creare tensione e orientare il giocatore.
- Ottimizzazione grafica e performance → miglioramenti visivi e tecnici per rendere l’esperienza più fluida e immersiva.

---

🖼️ Media


---

Video Demo
- link: https://www.youtube.com/watch?v=pcYNW_uPQRI

---

Download
- link:
