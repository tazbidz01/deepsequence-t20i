# CSE299.13: Junior Design Project Proposal

**Project Title:** DeepSequence-T20I: Contextual Batsman Vulnerability & Strategy Engine

### Student Information
*   **Name:** S M Tazbid Siddiqui  
    **ID:** 2321986042
*   **Name:** Mostofa Morshed  
    **ID:** 2321900042

**Faculty Advisor:** Muhammad Shafayat Osman (MUO)

---

## 1. Problem Statement
In short-format cricket like Twenty20 Internationals (T20Is), tactical demands switch rapidly between overs, meaning traditional flat averages (e.g., a batsman's overall average against spin) fail to inform live gameplay strategies. Current sports teams, academies, and performance analysts struggle with two main issues:
*   **The Fallacy of Aggregated Metrics:** Existing analytics software evaluates player data in static totals rather than chronological blocks. They overlook the *setup*—the way a tactical sequence of variations (e.g., three consecutive dot balls on a specific length) changes a batsman's shot risk on the following delivery.
*   **Manual Feature Correlation:** Performance analysts manually scrub text scorecards and match videos to link historical dismissals against high-pressure situational windows.

This project resolves these core problems by creating a dynamic, sequence-based analytical pipeline restricted specifically to T20I match data. It isolates tactical boundaries, maps physical ball trajectories from text feeds, and trains sequential deep learning architectures to explicitly uncover situational batsman weaknesses.

---

## 2. Introduction & Feasibility Study

### Introduction
DeepSequence-T20I is a full-stack data science platform that shifts cricket analytics from historical lookups to predictive sequence mapping. By standardizing high-granularity delivery rows, the platform analyzes the precise intersection of batting styles, bowling types, and tactical match phases.

### Feasibility Study
The operational viability of this application is grounded across three dimensions:
*   **Technical Feasibility:** Recurrent Neural Networks (LSTMs) with Attention layers are uniquely suited for short-format sequence prediction, parsing chronological series strings to identify tipping points in a batsman's behavioral profile.
*   **Data Feasibility:** The system relies entirely on open-source ball-by-ball T20I datasets provided by Cricsheet. Initial data verification successfully isolated massive historical transaction strings such as 17,088 continuous deliveries faced by a single baseline international player, proving the data volume is highly sufficient for sequence training steps.
*   **Operational Feasibility:** A 2-member engineering framework is ideal for this codebase. While one engineer addresses backend data modeling and NLP pipelines, the second implements relational storage tables and dashboard charting metrics concurrently.

---

## 3. Proposed Features and Technology

### Core Platform Features
*   **Granular Strike Rate & Dismissal Matrix:** Computes dynamic batting strike rates and dismissal metrics segmented strictly by T20I context variables: Match Phase (Powerplay: overs 0–6; Middle: overs 7–15; Death: overs 16–20), Bowler Hand (Left/Right), and Bowler Sub-Style (Pace, Off-spin, Leg-spin, Chinaman).
*   **NLP Commentary Feature Parser:** Employs an automated regular expression (Regex) indexing engine to process unstructured, text-based T20I commentary strings, transforming text lines into explicit physical variables (e.g., Line, Length, Shot Choice intent).
*   **Sequence-Based Weakness Predictor:** An LSTM network with an Attention mechanism that processes localized multi-parameter arrays across a rolling 6-to-12 delivery window to predict the exact probability of a batsman committing a fatal error or getting out on the upcoming ball.
*   **Automated Tactical PDF Generator:** Creates visual "Plan-of-Attack" cheat sheets detailing specific bowling vectors designed to exploit targeted opponent profiles.

### Technology Stack
*   **Data Processing & NLP:** Python, Pandas, NumPy, Regular Expressions (Regex)
*   **Machine Learning Pipelines:** PyTorch (LSTM + Attention Architecture), Scikit-Learn
*   **Storage Architecture:** SQLite / PostgreSQL (Relational Database)
*   **Frontend Dashboard View:** Streamlit (with Plotly/Matplotlib visualizations)
*   **PDF Generation:** ReportLab / FPDF

---

## 4. Weekly Plan (9-Week Schedule)
To satisfy the academic demand for rigorous weekly work tracking, the project operations are mapped out across a precise 9-week timeline with distributed milestones.

| Week | Phase Focus | Member 1 Tasks (Data Engineering & ML Core) | Member 2 Tasks (Full-Stack UI & Architecture) |
| :--- | :--- | :--- | :--- |
| **W1** | **Architecture & Schemas** | Define model sequence constraints, map training input dimensions, and document raw parameters. | Setup Git version tracking repository, design relational schemas, and wireframe dashboard layouts. |
| **W2** | **T20I Ingestion** | Build a batch parser engine to cleanly structure nested T20I JSON match deliveries. | Configure relational database storage instances and establish database connection schemas. |
| **W3** | **Feature Mapping Matrix** | Program the player attribute registry lookup logic to categorize bowler hands and sub-styles. | Build the backend utility paths to fetch processed delivery statistics. |
| **W4** | **Model Design & Sequence Setup** | Formulate multi-parameter input matrices (rolling 6-12 delivery windows) and design PyTorch LSTM layers. | Design the frontend dashboard structure and integrate interactive grid charts. |
| **W5** | **Model Training & Tuning** | Train the baseline sequence model and implement custom focal loss to handle dismissal class imbalances. | Establish backend routing pipelines to feed model parameters into the database layer. |
| **W6** | **NLP Layer & Commentary Parser** | Build regex parser rules in python to extract ball line, length, and shot intent from text commentary. | Implement the UI input interface for pasting, reviewing, and testing commentaries. |
| **W7** | **NLP to Model Pipeline** | Connect extracted commentary features into the sequence predictor model input matrix. | Connect frontend data visualization elements directly to the backend LSTM model endpoint. |
| **W8** | **PDF Cheat Sheet Generator** | Create functions to compile batsman vulnerability matrices and plot coordinates into a report buffer. | Develop the automated PDF report layout using ReportLab and add UI download triggers. |
| **W9** | **QA, Evaluation & Release** | Run model validation profiles checking consistency. Document model performance metrics and economic values. | Finalize UI styling and responsive themes. Package the codebase configurations and compile presentation slides. |

---

## 5. Software Lifecycle & Project Management

### A. Software Lifecycle
*   **Is it needed?**
    Yes. Modern T20 leagues (like BPL, IPL) require fast-paced tactical insights. Traditional metrics aggregate overall averages, masking localized sequence-based weaknesses (e.g., batting behavior following three consecutive dot balls). DeepSequence-T20I fills this gap, providing automated tactical assistance for high-pressure setups.
*   **Target Audience:**
    *   Professional Cricket Analysts (domestic/national teams, franchise setups).
    *   Coaching staff and private cricket academies wanting data-driven training paths.
    *   Sports broadcasters seeking sequence-based, real-time analytics to display on-screen.
*   **Is it possible?**
    Yes, technically and operationally feasible. Cricsheet provides complete, structured ball-by-ball datasets. LSTMs with attention are established architectures for temporal prediction, and python libraries like Streamlit and PyTorch enable rapid building and integration.

### B. Planning
*   **Project Leadership (PL) & Responsibilities:**
    *   **Member 1 (Data Engineering & ML Core):** Leads model constraint definitions, database mapping, feature lookup matrices, sequence construction, PyTorch training, and regex NLP parsing.
    *   **Member 2 (Full-Stack UI & Architecture):** Leads Git codebase tracking, relational DB schema configurations, REST-like backend utility development, Streamlit dashboard assembly, PDF generation formatting, and cross-browser stress tests.
*   **Feature / Page Mapping:**
    *   *Page 1: Batsman Profile Dashboard* (displays strike rates, matchups, and dismissal charts).
    *   *Page 2: Commentary Parsing Playground* (text input area where analysts paste commentaries to test NLP feature extraction).
    *   *Page 3: Live Sequence Simulator* (interface to queue up delivery variables and predict next-ball risks).
    *   *Page 4: PDF Report Hub* (interface to download compiled strategic "Plan-of-Attack" cheat sheets).

### C. Coding/Implementation
We will implement an **Agile Iterative coding cycle**. Python files will be organized cleanly in modular scripts:
*   `src/db.py`: Establishes DB connectors and runs creation schemas.
*   `src/parser.py`: Ingests and processes raw T20I JSON data files.
*   `src/nlp.py`: Cleans and extracts commentary features.
*   `src/model.py`: Builds and executes PyTorch LSTM predictions.
*   `src/report.py`: Assembles PDF layout blocks.
*   `src/app.py`: Glues components together inside a Streamlit application.

### D. Testing (Quality Assurance (QA))
*   **Unit & Integration Tests:** Implemented using `pytest` to run automated assertions against JSON parsing results and regular expression feature outputs.
*   **Deep Learning Evaluation:** Discarding flat accuracy metrics (due to rare dismissal events) in favor of optimizing **Precision**, **Recall**, and **F1-score** calculations.
*   **Data Consistency Checks:** Database query verification loops ensuring no null fields or mismatch records exist across table keys.

### E. Release and Update
*   **Version Control:** The repository will utilize Git branches to ensure seamless teamwork and code isolation.
*   **Data Refreshes:** The ingestion pipeline is designed dynamically; adding new T20I JSON files updates historical databases automatically.
*   **Model Retraining:** Future updates can ingest new season records to keep predictive boundaries sharp.
