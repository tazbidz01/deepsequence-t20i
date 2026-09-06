# DeepSequence T20I: Complete Codebase Walkthrough (Weeks 1 - 9)

*This document serves as your complete guide to defending your codebase. It maps every single task from your project proposal (Weeks 1 through 9) directly to the code that you and your partner wrote. You can use this to generate highly detailed slides or to easily answer any questions the professor asks during the demo.*

---

## Week 1: Architecture & Schemas

### Member 1: Data & ML Constraints
**Task:** Define model sequence constraints, map training input dimensions, and document raw parameters.
**Implementation:**
*   **Sequence Constraints (`src/config.py`):** Configured `SEQUENCE_LENGTH = 12` to map the rolling window of a 2-over period. Also set `PADDING_VALUE = 0.0` for batsmen who have faced fewer than 12 deliveries.
*   **Input Dimensions:** Mapped categorical context parameters (like `MATCH_PHASES` and `BOWLER_STYLES`) to integer maps. Configured 18 embedding channels and 7 continuous features to create the final 25-dimensional input vector for the neural network.
*   **Raw Parameters (`src/train.py`):** Configured loss constraints including `focal_alpha = 0.92` to prioritize rare dismissal events over safe deliveries.

### Member 2: UI & Arch Setup
**Task:** Setup Git version tracking repository, design relational schemas, and wireframe dashboard layouts.
**Implementation:**
*   **Git Setup (`.gitignore`):** excluded raw JSONs, processed `*.db` files, and `__pycache__` to prevent bloating the repository.
*   **Relational Schemas (`src/db.py`):** Designed the core SQLite schema, initializing the `matches`, `players`, and `deliveries` tables.
*   **Dashboard Wireframes (`src/app.py`):** Sketched out the initial Streamlit framework and configured the dark mode CSS (`#0F172A` backgrounds with `#38BDF8` accents).

---

## Week 2: T20I Ingestion

### Member 1: Batch Parser Engine
**Task:** Build a batch parser engine to cleanly structure nested T20I JSON match deliveries.
**Implementation:**
*   **JSON Parsing (`src/parser.py`):** Built `parse_cricsheet_json()` to dive layer-by-layer into the nested Cricsheet trees, extracting match metadata and flattening the ball-by-ball dictionaries.
*   **Batch Processing:** Created `ingest_all_jsons()` to iterate over the `data/raw/` directory, aggregating millions of records into Pandas DataFrames and pushing them to the database using `.to_sql()`.

### Member 2: Relational Database Storage
**Task:** Configure relational database storage instances and establish database connection schemas.
**Implementation:**
*   **Database Connections (`src/db.py`):** Created the `get_connection()` path to ensure safe, repeatable connections to `t20i_engine.db`.
*   **Foreign Keys:** Enforced referential integrity in the SQLite schema by linking `deliveries` back to `matches` and `players` using primary/foreign key relationships.
*   **Automated Updates (`src/update_dataset.py`):** Engineered a weekly cron job script to safely trigger automatic ZIP downloads and database appends without crashing the active UI.

---

## Week 3: Feature Mapping

### Member 1: Player Attribute Registry Lookup
**Task:** Program player attribute registry lookup logic to categorize bowler hands and sub-styles.
**Implementation:**
*   **Phase Mathematics (`src/utils.py`):** Programmed advanced SQL `CASE` statements to dynamically group balls into strict Match Phases (Powerplay: Overs 0-5, Middle: Overs 6-14, Death: Overs 15-19).
*   **SQL Joins:** Built backend SQL logic to dynamically compute strike rates strictly based on these extracted phases rather than flat averages.

### Member 2: Backend Utility Paths & Scrapers
**Task:** Build the backend utility paths to fetch processed delivery statistics.
**Implementation:**
*   **Utility APIs (`src/utils.py`):** Centralized all database queries into reusable functions like `get_batsman_kpis()`.
*   **Hybrid Cloudscraper (`src/fetch_styles.py`):** Built a web scraper to fetch missing Bowler Styles from ESPNcricinfo, utilizing a BeautifulSoup HTML DOM parser and an offline CSV fallback to bypass Cloudflare bot protection.

---

## Week 4: Model Design

### Member 1: Multi-Parameter Input Matrices
**Task:** Formulate multi-parameter input matrices and design PyTorch LSTM layers.
**Implementation:**
*   **LSTM Design (`src/model.py`):** Built the `DeepSequenceModel` class using PyTorch (`nn.Module`). Configured a dynamic LSTM layer (`batch_first=True`) and attached a Sigmoid activation function to squash predictions into a clean percentage (0 to 1).
*   **Tensor Processing (`src/features.py`):** Developed `SequencePreprocessor` to normalize runs and stack the arrays into a massive 56-Dimensional Tensor for the ML model.

### Member 2: Frontend Dashboard Structure
**Task:** Design the frontend dashboard structure and integrate interactive grid charts.
**Implementation:**
*   **Dynamic UI Rendering (`src/app.py`):** Programmed a Streamlit slider allowing the user to select any sequence length from 3 to 12.
*   **Interactive Grids:** Used a `for` loop to dynamically render the exact number of column dropdown menus required to match the user's selected sequence length.

---

## Week 5: Model Training

### Member 1: Training & Focal Loss
**Task:** Train the baseline sequence model and implement custom focal loss to handle dismissal class imbalances.
**Implementation:**
*   **Custom Focal Loss (`src/train.py`):** Wrote a custom PyTorch loss module that exponentially penalizes the network for predicting "Safe" when a rare dismissal actually occurs, forcing the network to learn high-risk sequences.
*   **Training Loop:** Initialized the Adam optimizer and ran the dataset through the forward/backward passes for 10 epochs, saving the final `.pth` weights.

### Member 2: Backend Routing Pipelines
**Task:** Establish backend routing pipelines to feed model parameters into the database layer.
**Implementation:**
*   **Model Registry (`src/db.py`):** Added a `model_registry` table to SQLite.
*   **Training Logs (`src/utils.py`):** Built `log_model_training()` to automatically record the final loss metrics and file paths of newly trained `.pth` files into the database.

---

## Week 6: NLP Commentary

### Member 1: Regex Parser Rules
**Task:** Build regex parser rules in python to extract ball line, length, and shot intent from text commentary.
**Implementation:**
*   **NLP Extraction Engine:** Designed complex Python regular expressions (`re` module) to scan unstructured human commentary. 
*   **Categorization:** Built logic to hunt for keywords like "yorker", "banged in short", or "wide outside off" and convert them into rigid categorical strings (`length`, `line`).

### Member 2: UI Input Interface
**Task:** Implement the UI input interface for pasting, reviewing, and testing commentaries.
**Implementation:**
*   **NLP Dashboard Tab (`src/app.py`):** Created Tab 4 (Commentary NLP Parser) using `st.text_area` to allow coaches to paste in live text from Cricinfo.
*   **Live Testing:** Added visual feedback containers to instantly display the parsed Line and Length results directly on the screen for review.

---

## Week 7: NLP to Model Pipeline

### Member 1: Sequence Predictor Model Input Matrix
**Task:** Connect extracted commentary features into the sequence predictor model input matrix.
**Implementation:**
*   **Matrix Injection (`src/features.py`):** Updated the `SequencePreprocessor` to accept the newly parsed NLP categories. Used `OneHotEncoder` from `scikit-learn` to mathematically transform the parsed words (e.g., "yorker") into binary arrays so the PyTorch tensor could read them.

### Member 2: Frontend Data Visualization
**Task:** Connect frontend data visualization elements directly to the backend LSTM model endpoint.
**Implementation:**
*   **Live Forward Pass (`src/app.py`):** Connected the "Predict Vulnerability" button to the PyTorch backend. Implemented `torch.no_grad()` inside the Streamlit loop to safely execute a live prediction on the user's 12-ball input and output the result to a visual progress bar gauge.

---

## Week 8: PDF Report Gen (Note: Adapted to UI Analytics)

*Note: Since standard PDF generation was pivoted in favor of real-time dashboard analytics, here is how the team fulfilled the reporting requirements.*

### Member 1: Vulnerability Matrices
**Task:** Create functions to compile batsman vulnerability matrices and plot coordinates into a report buffer.
**Implementation:**
*   **Historical Vulnerability (`src/utils.py`):** Built advanced SQL aggregations to calculate exact Strike Rates and Dismissal frequencies segmented strictly by the Bowler's sub-style (e.g., Left-Arm Pace). This generates the raw matrices required for the frontend heatmaps.

### Member 2: Automated Report Layout
**Task:** Develop the automated report layout and add UI triggers.
**Implementation:**
*   **Analytics Layout (`src/app.py`):** Designed Tab 1 (Batsman Profile) to display these matrices in beautiful, split-bar column layouts. Engineered `@st.cache_data` memory hashing to ensure these massive matrix reports load instantly in the browser without freezing.

---

## Week 9: QA & Release

### Member 1: Validation Profiles
**Task:** Run model validation profiles checking consistency. Document model performance metrics and economic values.
**Implementation:**
*   **Validation Metrics:** Generated the Precision, Recall, and F1-Score metrics to prove the Focal Loss resolved the class imbalances.
*   **Economic Value:** Documented the exact server hosting costs ($20/month on AWS) versus traditional data firm expenses to prove the tool's massive ROI.

### Member 2: QA, Styling, and Presentation
**Task:** Finalize UI styling and responsive themes. Package the codebase configurations and compile presentation slides.
**Implementation:**
*   **UI Polish (`src/app.py`):** Finalized the responsive layouts, sidebar controllers, and dark mode theming to ensure the app looks professional on any coach's laptop.
*   **Presentation Prep:** Finalized the `.gitignore`, ensured the SQLite `.db` file initializes cleanly for new users, and generated the official `DeepSequence_Final_Presentation_V3.pptx` slide deck based on the finalized architecture.
