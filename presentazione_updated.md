# MVTec Anomaly Benchmark 
# Una pipeline completa per la visual anomaly detection 

### Abstract

Questo progetto realizza una pipeline end-to-end per **visual anomaly detection** (VAD) su **MVTec AD**, includendo: training/evaluation con **Anomalib**, inferenza su singola immagine e una **interfaccia interattiva** per sperimentazione (upload, disegno difetti, confronto tra modelli, consultazione metriche). Il documento descrive formulazione del problema, principi dei modelli e trade-off ingegneristici (accuratezza, latenza, memoria, deployability).



## Sommario

1. [Motivazioni](#1-motivazioni)
   - 1.1 [Contesto Applicativo](#11-contesto-applicativo)
   - 1.2 [Paradigma One-Class Classification](#12-paradigma-one-class-classification)
   - 1.3 [Formalizzazione del Problema](#13-formalizzazione-del-problema-detection--localization)
   - 1.4 [Trend Recenti e Posizionamento](#14-trend-recenti-20242026-e-posizionamento-del-progetto)
   - 1.5 [Obiettivi del Progetto](#15-obiettivi-del-progetto)
2. [Il Dataset MVTec AD](#2-il-dataset-mvtec-ad)
3. [Architettura del Sistema](#3-architettura-del-sistema)
4. [Modelli Implementati](#4-modelli-implementati)
   - 4.1 [PatchCore](#41-patchcore)
   - 4.2 [PaDiM](#42-padim)
   - 4.3 [FastFlow](#43-fastflow)
   - 4.4 [STFPM](#44-stfpm)
   - 4.5 [EfficientAD](#45-efficientad)
5. [Implementazione Tecnica](#5-implementazione-tecnica)
   - 5.1 [Pipeline di Training](#51-pipeline-di-training)
   - 5.2 [Pipeline di Inference](#52-pipeline-di-inference)
   - 5.3 [Interfaccia Interattiva](#53-interfaccia-interattiva)
6. [Metriche di Valutazione](#6-metriche-di-valutazione)
7. [Conclusioni](#7-conclusioni)

---

## 1. Motivazioni

### 1.1 Contesto Applicativo

La **rilevazione di anomalie** (*Anomaly Detection*) rappresenta una delle sfide più rilevanti nell'ambito dell'apprendimento automatico applicato al **controllo qualità industriale**. In **contesti manifatturieri**, la capacità di identificare automaticamente difetti in prodotti, componenti o materiali consente di:

- Ridurre i costi associati a ispezioni manuali
- Aumentare la velocità di produzione mantenendo elevati standard qualitativi
- Garantire consistenza e ripetibilità del processo di controllo qualità
- Rilevare difetti sottili che potrebbero sfuggire all'ispezione umana

### 1.2 Paradigma One-Class Classification

Un aspetto fondamentale della rilevazione di anomalie industriali è che tipicamente si dispone esclusivamente di **campioni normali** durante la fase di addestramento. Questo scenario, noto come *One-Class Classification* o *Unsupervised Anomaly Detection*, presenta sfide peculiari:

- **Sbilanciamento intrinseco**: I campioni anomali sono rari per definizione
- **Varietà delle anomalie**: I difetti possono manifestarsi in forme non prevedibili a priori
- **Assenza di supervisione diretta**: Non si dispone di etichette per le anomalie durante il training

I modelli implementati in questo progetto adottano **strategie diverse per apprendere la distribuzione dei dati normali e identificare deviazioni da essa durante l'inferenza**.

### 1.3 Formalizzazione del Problema (Detection + Localization)

Nel contesto della **visual anomaly detection** su immagini, si consideri un'immagine $I \in \mathbb{R}^{H\times W\times 3}$. Un modello produce tipicamente:

- uno **score globale** $s(I) \in \mathbb{R}$ (o in $[0,1]$ dopo normalizzazione), usato per decidere se l'immagine è anomala;
- una **mappa di anomalia** $A(I) \in [0,1]^{H\times W}$, usata per localizzare le regioni responsabili.

La decisione binaria è data da una soglia $\tau$:

$$
\hat{y}(I) = \mathbb{1}[s(I) > \tau]
$$

Analogamente, per la segmentazione si può thresholdare la mappa con $\tau_p$:

$$
\hat{M}(I) = \mathbb{1}[A(I) > \tau_p]
$$

Nei metodi one-class la rete non "impara il difetto", ma costruisce un riferimento del *normale* (memoria, distribuzione, likelihood o distillation). Le anomalie emergono come deviazioni rispetto a quel riferimento.

### 1.4 Trend Recenti (2024-2026) e Posizionamento del Progetto

Nella letteratura e nelle applicazioni industriali recenti si osservano alcune direttrici evolutive utili per inquadrare questo lavoro:

- **Backbone sempre più forti**: crescente adozione di feature extractor pre-addestrati (self-supervised e/o Transformer) come base per metodi one-class; questo rende ancora più importanti pipeline modulari e configurabili.
- **Focus su latenza e deploy**: oltre all'accuratezza, diventano centrali throughput, footprint e compatibilità con tool di deployment (quantizzazione/esportazione, accelerazione hardware).
- **Interpretabilità operativa**: heatmap, diagnostica delle distribuzioni degli score e strumenti per calibrare soglie sono cruciali per l'adozione reale.
- **Anomalie "logiche"**: non tutte le anomalie sono texture/locali; alcuni difetti emergono da combinazioni globali inconsistenti. Approcci come EfficientAD evidenziano questa esigenza.

Il progetto seleziona cinque famiglie rappresentative (memory-based, distribuzionali, likelihood-based, distillation, efficient real-time) e le unifica in una pipeline unica, adatta sia a benchmarking sia a dimostrazione interattiva.

### 1.5 Obiettivi del Progetto

Il presente progetto si propone di:

1. **Implementare una pipeline end-to-end** per il training, la valutazione e l'inferenza di modelli di anomaly detection
2. **Confrontare sistematicamente** cinque architetture allo stato dell'arte
3. **Fornire un'interfaccia utente interattiva** per la sperimentazione e la visualizzazione dei risultati
4. **Documentare e rendere riproducibili** tutti gli esperimenti condotti

---

## 2. Il Dataset MVTec AD

### 2.1 Descrizione Generale

Il **MVTec Anomaly Detection Dataset** (MVTec AD) costituisce il benchmark standard de facto per la valutazione di algoritmi di anomaly detection in ambito industriale. Pubblicato da MVTec Software GmbH, il dataset comprende immagini ad alta risoluzione di 15 categorie distinte, suddivise in due macro-classi:

#### Categorie di Texture (5)
| Categoria | Descrizione | Tipi di Difetto |
|-----------|-------------|-----------------|
| **Carpet** | Tessuto per tappeti | color, cut, hole, metal contamination |
| **Grid** | Griglia metallica | bent, broken, glue, metal contamination |
| **Leather** | Pelle | color, cut, fold, glue |
| **Tile** | Piastrella | crack, glue strip, gray stroke, oil |
| **Wood** | Legno | color, combined, hole, liquid |

#### Categorie di Oggetti (10)
| Categoria | Descrizione | Tipi di Difetto |
|-----------|-------------|-----------------|
| **Bottle** | Bottiglia | broken_large, broken_small, contamination |
| **Cable** | Cavo elettrico | bent_wire, cable_swap, combined, cut_inner_insulation |
| **Capsule** | Capsula farmaceutica | crack, faulty_imprint, poke, scratch |
| **Hazelnut** | Nocciola | crack, cut, hole, print |
| **Metal Nut** | Dado metallico | bent, color, flip, scratch |
| **Pill** | Pillola | color, combined, contamination, crack |
| **Screw** | Vite | manipulated_front, scratch_head, scratch_neck, thread_side |
| **Toothbrush** | Spazzolino | defective |
| **Transistor** | Transistor | bent_lead, cut_lead, damaged_case, misplaced |
| **Zipper** | Cerniera | broken_teeth, combined, fabric_border, fabric_interior |

### 2.2 Caratteristiche Statistiche

- **Dimensione**: ordine di grandezza ~5 GB
- **Cardinalità**: **oltre 5000 immagini** ad alta risoluzione complessive (15 categorie)
- **Risoluzione**: variabile per categoria, tipicamente nell'ordine di centinaia di pixel per lato
- **Training set**: esclusivamente immagini *good* (normali)
- **Test set**: immagini *good* + immagini con difetti; per i difetti sono disponibili **annotazioni pixel-accurate** (ground truth) per la localizzazione

### 2.3 Perché MVTec AD

MVTec AD è stato scelto perché rappresenta il benchmark standard de facto nella comunità di visual anomaly detection, garantendo comparabilità diretta con la letteratura esistente. Alternative come VisA, BTAD o DAGM esistono, ma MVTec AD rimane il riferimento più utilizzato per validare nuovi approcci.

---

## 3. Architettura del Sistema

### 3.1 Design Philosophy

L'architettura del sistema è stata progettata seguendo i principi di:

- **Modularità**: Separazione netta tra componenti funzionali (training, inference, UI)
- **Configurabilità**: Parametrizzazione completa tramite file YAML
- **Riproducibilità**: Gestione deterministica degli esperimenti
- **Estensibilità**: Facilità nell'aggiunta di nuovi modelli

### 3.2 Dipendenza da Anomalib

Il progetto si basa su **Anomalib**, una libreria open-source focalizzata su visual anomaly detection che fornisce un'API modulare e pipeline standardizzate, con sviluppo continuo nella community open-source.

### 3.3 Stack Tecnologico

```
┌─────────────────────────────────────────────────┐
│              Interfaccia Interattiva            │
├─────────────────────────────────────────────────┤
│              Application Layer                  │
│    (app.py, inference.py, train.py)             │
├─────────────────────────────────────────────────┤
│                Core Module                      │
│    (config.py, models.py, utils.py)             │
├─────────────────────────────────────────────────┤
│                 Anomalib                        │
├─────────────────────────────────────────────────┤
│     PyTorch (v2.0+) / PyTorch Lightning         │
├─────────────────────────────────────────────────┤
│               CUDA / cuDNN                      │
└─────────────────────────────────────────────────┘
```

---

## 4. Modelli Implementati

### 4.1 PatchCore

#### Riferimento Bibliografico
> Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022). *Towards Total Recall in Industrial Anomaly Detection*. CVPR 2022.

#### Architettura e Principio di Funzionamento

PatchCore è un metodo **memory-based** che non richiede training iterativo. L'algoritmo si articola nelle seguenti fasi:

**1. Estrazione delle Feature**

PatchCore utilizza un backbone CNN pre-addestrato su ImageNet (nel progetto: `WideResNet-50-2`) per estrarre feature a livello di patch. Le feature vengono estratte da layer intermedi della rete (`layer2`, `layer3`), consentendo di catturare sia informazioni semantiche locali che più astratte.

**2. Costruzione della Memory Bank**

Per ogni immagine di training, viene costruita una memoria di feature normali. Data la potenziale esplosione dimensionale, PatchCore applica **Coreset Subsampling** (selezione tipo k-center greedy) che garantisce:
- Riduzione significativa della memoria (tipicamente 10x)
- Preservazione della rappresentatività delle feature normali

**3. Anomaly Scoring**

Durante l'inferenza, per ogni patch dell'immagine di test si calcola la **distanza dal nearest neighbor nella memory bank**. Lo score a livello di immagine è l'aggregazione degli score locali, mentre la **mappa di anomalia** si ottiene riportando gli score patch-wise allo spazio immagine.

Un dettaglio importante è che PatchCore può rendere lo score image-level più robusto con un *re-weighting density-aware*, attenuando i falsi allarmi quando il nearest neighbor è un punto "isolato" nel memory bank. Dal punto di vista ingegneristico, il collo di bottiglia può diventare la ricerca kNN su molte patch: tecniche come FAISS permettono di accelerare significativamente questa operazione.

#### Punti di Forza e Limitazioni

| PUNTI DI FORZA | LIMITAZIONI |
|----------------|-------------|
| Nessun training iterativo richiesto | Memoria proporzionale al dataset |
| Alta accuratezza su texture | Velocità di inferenza dipende dalla dimensione della memory bank |
| Funziona con pochi campioni normali | Sensibile alla qualità del backbone |

---

### 4.2 PaDiM (Patch Distribution Modeling)

#### Riferimento Bibliografico
> Defard, T., Setkov, A., Loesch, A., & Audigier, R. (2021). *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. ICPR 2021.

#### Architettura e Principio di Funzionamento

PaDiM modella la **distribuzione statistica** delle feature normali a ogni posizione spaziale della mappa delle feature.

**1. Estrazione Multi-scala delle Feature**

PaDiM concatena feature da multiple layer (`layer1`, `layer2`, `layer3`) per ottenere una rappresentazione multi-scala che cattura informazioni a diverse scale spaziali.

**2. Modellazione della Distribuzione**

Per ogni posizione $(i, j)$ nella mappa delle feature, PaDiM stima una **distribuzione Gaussiana multivariata**:

$$
\mathbf{x}_{ij} \sim \mathcal{N}(\boldsymbol{\mu}_{ij}, \boldsymbol{\Sigma}_{ij})
$$

Per ridurre dimensionalità e costo computazionale, si applica feature subsampling e regolarizzazione della covarianza per stabilità numerica.

**3. Anomaly Scoring tramite Distanza di Mahalanobis**

Lo score di anomalia è calcolato come:

$$
M(\mathbf{x}_{ij}) = \sqrt{(\mathbf{x}_{ij} - \boldsymbol{\mu}_{ij})^T \boldsymbol{\Sigma}_{ij}^{-1} (\mathbf{x}_{ij} - \boldsymbol{\mu}_{ij})}
$$

La distanza di Mahalanobis tiene conto della correlazione tra le dimensioni delle feature ed è invariante rispetto a trasformazioni affini dello spazio.

PaDiM è un modello **location-aware**: la Gaussiana è stimata per ciascuna posizione. Questo è un vantaggio su oggetti ben allineati (scenario industriale tipico), ma può diventare fragile se variano molto pose, traslazioni o rotazioni.

---

### 4.3 FastFlow

#### Riferimento Bibliografico
> Yu, J., et al. (2021). *FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows*. arXiv:2111.07677.

#### Architettura e Principio di Funzionamento

FastFlow utilizza **Normalizing Flows** per modellare la distribuzione delle feature normali con una formulazione probabilistica basata sulla stima della likelihood.

**1. Concetto di Normalizing Flow**

Un normalizing flow definisce una trasformazione invertibile $f: \mathcal{Z} \rightarrow \mathcal{X}$ che mappa una distribuzione semplice (base) $p_Z(\mathbf{z})$ in una distribuzione complessa $p_X(\mathbf{x})$:

$$
p_X(\mathbf{x}) = p_Z(f^{-1}(\mathbf{x})) \cdot \left| \det \frac{\partial f^{-1}}{\partial \mathbf{x}} \right|
$$

**2. Architettura 2D**

FastFlow implementa un flow 2D che preserva la struttura spaziale delle feature map, consentendo di derivare naturalmente una anomaly map calcolando il contributo di NLL per ogni location.

**3. Training e Anomaly Scoring**

L'anomaly score è dato dalla **negative log-likelihood**: feature con bassa likelihood (alto score) sono considerate anomale.

La stima di densità tramite flow sostituisce del tutto la memory bank: una volta addestrato, il costo in inferenza non dipende dal numero di campioni di training. Il rovescio della medaglia è che qui c'è un **vero training** con iperparametri e rischio di overfitting.

---

### 4.4 STFPM (Student-Teacher Feature Pyramid Matching)

#### Riferimento Bibliografico
> Wang, G., et al. (2021). *Student-Teacher Feature Pyramid Matching for Anomaly Detection*. arXiv:2103.04257.

#### Architettura e Principio di Funzionamento

STFPM implementa un paradigma **Student-Teacher** basato sull'idea che uno studente addestrato a imitare un teacher su dati normali presenterà discrepanze quando esposto ad anomalie.

**1. Teacher Network**

Il teacher è una CNN pre-addestrata (frozen) che estrae feature a multiple scale formando una **Feature Pyramid**.

**2. Student Network**

Lo studente ha architettura identica al teacher, ma è inizializzato casualmente e addestrato a riprodurre le feature del teacher sui dati normali:

$$
\mathcal{L} = \sum_{l \in \text{layers}} \left\| F_T^{(l)}(\mathbf{x}) - F_S^{(l)}(\mathbf{x}) \right\|_2^2
$$

**3. Anomaly Detection**

Le anomalie vengono rilevate misurando la **discrepanza** tra teacher e student:
- Su dati normali: bassa discrepanza (lo studente ha imparato a imitare)
- Su anomalie: alta discrepanza (lo studente non sa come rispondere)

L'uso di feature multi-scala consente di rilevare difetti a grana fine (layer1), media scala (layer2) e strutturali (layer3).

---

### 4.5 EfficientAD

#### Riferimento Bibliografico
> Batzner, K., Heckler, L., & König, R. (2024). *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies*. WACV 2024.

#### Architettura e Principio di Funzionamento

EfficientAD è progettato per ottenere **latenze dell'ordine dei millisecondi** mantenendo accuratezza competitiva, rendendolo adatto a scenari real-time.

EfficientAD combina:
- Un **feature extractor leggero** (pensato per throughput elevato)
- Un impianto **student-teacher** per anomalie strutturali/locali
- Un **autoencoder** per catturare **anomalie logiche** (configurazioni globalmente inconsistenti)

Il punto distintivo è questa separazione: lo student-teacher intercetta bene anomalie locali, mentre l'autoencoder è utile quando localmente tutto sembra plausibile ma la configurazione globale è errata (es. orientamento o ordine di componenti).

Dal lato pratico, EfficientAD richiede alcune accortezze specifiche: batch size molto piccolo, trasformazioni dedicate e la preparazione di un piccolo dataset ImageNet-like per il training.

---

## 5. Implementazione Tecnica

Il sistema è organizzato in tre componenti principali che collaborano per garantire modularità, riproducibilità e facilità d'uso:

- **Modulo di configurazione** (`config.py`): definisce le categorie disponibili, i percorsi delle directory e le configurazioni YAML per ciascun modello.
- **Modulo modelli** (`models.py`): gestisce l'istanziazione dei modelli e il recupero dei checkpoint.
- **Modulo utilità** (`utils.py`): contiene le operazioni necessarie per rendere comparabili e visualizzabili gli output dei diversi metodi.

### 5.1 Pipeline di Training

La pipeline di training segue un flusso standardizzato che si applica uniformemente a tutti i modelli. Per ciascuna categoria del dataset MVTec AD, il sistema:

1. **Carica la configurazione** del modello da file YAML, permettendo di modificare iperparametri senza toccare il codice.
2. **Inizializza il DataModule** MVTec AD tramite Anomalib, che gestisce automaticamente il caricamento dei dati, le trasformazioni e la suddivisione train/test.
3. **Istanzia il modello** secondo i parametri specificati nella configurazione.
4. **Esegue il training** (quando previsto dal metodo) utilizzando PyTorch Lightning, che standardizza il loop di addestramento e la gestione dei checkpoint.
5. **Valuta le performance** sul test set, calcolando metriche sia a livello di immagine che a livello di pixel.

Al termine, il sistema salva automaticamente i pesi del modello e le metriche in formato JSON, garantendo riproducibilità e facilitando il confronto tra esperimenti.

### 5.2 Pipeline di Inference

L'inferenza è progettata per essere semplice e "a prova di demo": dato un input (immagine) e una scelta (modello, categoria), il sistema carica il checkpoint corretto, esegue la predizione e restituisce sia uno **score globale** sia una **mappa di anomalia** per spiegare *dove* si trova il potenziale difetto.

L'output dell'inferenza viene poi trasformato in una **visualizzazione composita** che include:

1. **Immagine originale**: per riferimento visivo.
2. **Heatmap delle anomalie**: rappresentazione colorimetrica dello score locale, dove colori caldi indicano alta probabilità di anomalia.
3. **Overlay**: sovrapposizione della heatmap sull'immagine originale per evidenziare la corrispondenza spaziale.
4. **Maschera predetta**: contorni che delimitano le regioni classificate come anomale.

Questa visualizzazione multi-pannello è cruciale per l'interpretabilità: permette di capire non solo *se* il modello rileva un'anomalia, ma anche *dove* e *con quale confidenza*.

Un aspetto tecnico rilevante riguarda la **normalizzazione delle heatmap**: i diversi modelli producono anomaly map con range di valori differenti. Per garantire visualizzazioni comparabili, il sistema implementa una normalizzazione adattiva che preserva l'ordinamento relativo dei valori e gestisce correttamente artefatti come il padding.

### 5.3 Interfaccia Interattiva

Un aspetto distintivo del progetto è l'interfaccia utente interattiva. Nel contesto dell'anomaly detection industriale, un'interfaccia di questo tipo non è un "accessorio" ma uno strumento essenziale per diverse ragioni.

#### Perché un'interfaccia è significativa

**1. Validazione qualitativa**

Le metriche aggregate (AUROC, F1) possono nascondere comportamenti problematici del modello. Solo ispezionando visivamente le heatmap si può verificare se il modello sta realmente "capendo" il problema o se sta sfruttando shortcut statistici (es. artefatti del background invece del difetto effettivo).

**2. Comunicazione con stakeholder non tecnici**

In contesto industriale, i decision-maker spesso non sono esperti di machine learning. Un'interfaccia che permette di caricare un'immagine e vedere immediatamente il risultato è molto più efficace di una tabella di metriche per dimostrare il valore del sistema e costruire fiducia nella tecnologia.

**3. Debugging e analisi degli errori**

Poter testare rapidamente casi specifici (immagini problematiche, edge case, condizioni di illuminazione particolari) accelera enormemente il ciclo di sviluppo e miglioramento del modello. Un ricercatore può identificare pattern di fallimento in minuti invece che in ore.

**4. Confronto controllato tra modelli**

Visualizzare side-by-side le risposte di diversi modelli sulla stessa immagine rende immediatamente evidenti differenze che sarebbero difficili da cogliere confrontando solo numeri. Questo tipo di confronto controllato è fondamentale per decisioni informate sulla scelta del modello.

#### Funzionalità implementate

L'interfaccia è organizzata in moduli tematici, ciascuno pensato per un caso d'uso specifico:

**Upload e Analisi**: il caso d'uso più diretto. L'utente carica un'immagine, seleziona modello e categoria, e ottiene la visualizzazione completa con heatmap e score. Questo simula il workflow operativo reale in uno scenario di controllo qualità.

**Disegno Difetti**: uno strumento didattico e di stress-test particolarmente interessante. Permette di disegnare difetti artificiali su immagini normali e verificare se il modello li rileva. Questo consente di:
- Testare la sensibilità del modello a perturbazioni locali
- Verificare la coerenza spaziale della heatmap (l'hotspot segue il difetto disegnato?)
- Identificare eventuali falsi positivi in aree non modificate

**Confronto Modelli**: presenta le risposte di più modelli sulla stessa immagine, in una visualizzazione affiancata. Questo rende il benchmarking tangibile e comprensibile anche senza analizzare tabelle di metriche.

**Consultazione Metriche**: aggrega e presenta le metriche di performance calcolate durante il training, permettendo di navigare i risultati per modello e categoria.

---

## 6. Metriche di Valutazione

### 6.1 Metriche di Efficacia

#### AUROC (Area Under ROC Curve)

L'AUROC misura la capacità discriminativa del modello nel distinguere campioni normali da anomali:

$$
\text{AUROC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt
$$

dove:
- **TPR** (True Positive Rate): Proporzione di anomalie correttamente identificate
- **FPR** (False Positive Rate): Proporzione di campioni normali erroneamente classificati come anomali

Il progetto calcola l'AUROC a due livelli:
- **Image AUROC**: Classificazione a livello di immagine (anomala vs normale)
- **Pixel AUROC**: Segmentazione a livello di pixel (localizzazione del difetto)

#### F1-Score

L'F1-Score è la media armonica di precisione e recall:

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

A differenza dell'AUROC che valuta il ranking, l'F1-Score richiede la scelta di una soglia operativa, rendendolo più rilevante per applicazioni reali.

### 6.2 Metriche di Efficienza

| Metrica | Descrizione | Unità |
|---------|-------------|-------|
| **Train Time** | Tempo totale di addestramento | secondi |
| **Inference Time** | Tempo per processare una singola immagine | millisecondi |
| **FPS** | Frame per secondo durante inference | frame/s |
| **Model Size** | Dimensione del checkpoint su disco | MB |

In applicazioni industriali e in deployment (edge o real-time), le metriche di efficienza sono spesso decisive quanto quelle di accuratezza. Un modello leggermente meno accurato ma 10 volte più veloce può essere preferibile in contesti con vincoli di latenza stringenti.

---

## 7. Conclusioni

### 7.1 Contributi del Progetto

Il presente progetto fornisce:

1. **Pipeline completa e modulare** per anomaly detection industriale
2. **Implementazione e confronto sistematico** di 5 architetture allo stato dell'arte
3. **Interfaccia utente interattiva** per sperimentazione e visualizzazione
4. **Codice riproducibile** con configurazioni esternalizzate

### 7.2 Trade-off tra i Modelli

| Modello | Idea chiave | Training iterativo | Memoria | Latenza attesa | Nota pratica |
|---------|------------|-------------------|---------|----------------|-------------|
| **PatchCore** | kNN su memory bank di patch | No (fit della memoria) | Alta-Media | Media | Ottimo baseline, molto competitivo su texture |
| **PaDiM** | Gaussiana per patch + Mahalanobis | No (stima statistiche) | Media-Bassa | Alta (rapida) | Buon compromesso, stabile se ben regolarizzato |
| **FastFlow** | Normalizing flow (likelihood) | Sì | Bassa | Alta | Approccio probabilistico, buona deployability |
| **STFPM** | Distillation student-teacher multi-scala | Sì | Bassa | Alta | Molto interpretabile via mappe di errore |
| **EfficientAD** | Distillation + componenti leggere | Sì | Bassa | Molto alta | Pensato per throughput elevato e casi real-time |

### 7.3 Linee Guida per la Scelta del Modello

- **Massima accuratezza**: PatchCore (texture) o STFPM (strutturale)
- **Tempo reale**: EfficientAD o FastFlow
- **Memoria limitata**: PaDiM o EfficientAD
- **Pochi campioni di training**: PatchCore
- **Balance generale**: FastFlow

### 7.4 Possibili Estensioni Future

- Integrazione di modelli aggiuntivi (Reverse Distillation, CFlow-AD)
- Supporto per altri dataset (BTAD, VisA, DAGM)
- Ottimizzazione per edge deployment (quantizzazione, pruning)
- Ensemble di modelli per robustezza

### 7.5 Sintesi dei Punti Chiave

- L'anomaly detection one-class richiede metodi che apprendano il *normale* (memoria, distribuzione, likelihood, distillation) e misurino la deviazione in inferenza.
- La localizzazione (anomaly map) è centrale per interpretabilità e per l'uso pratico in controllo qualità.
- La comparazione tra modelli evidenzia trade-off reali tra accuratezza, latenza e memoria.
- La struttura modulare rende gli esperimenti riproducibili e il sistema estensibile.
- L'interfaccia interattiva non è accessoria ma essenziale per validazione qualitativa, comunicazione e debugging.

---

## Riferimenti Bibliografici

1. Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). *MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR 2019.

2. Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021). *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. IJCV 2021.

3. Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022). *Towards Total Recall in Industrial Anomaly Detection*. CVPR 2022.

4. Defard, T., Setkov, A., Loesch, A., & Audigier, R. (2021). *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. ICPR 2021.

5. Yu, J., et al. (2021). *FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows*. arXiv:2111.07677.

6. Wang, G., et al. (2021). *Student-Teacher Feature Pyramid Matching for Anomaly Detection*. arXiv:2103.04257.

7. Batzner, K., Heckler, L., & König, R. (2024). *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies*. WACV 2024.

8. Akcay, S., et al. (2022). *Anomalib: A Deep Learning Library for Anomaly Detection*. ICIP 2022.

---
