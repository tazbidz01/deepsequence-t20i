# DeepSequence T20I: System Flowchart (V2)

Here is the updated flowchart representing the new features of the project (including the Cron Job, NLP processing, 56-D Tensor, and 12-Ball input). 

You can take a screenshot of this diagram and insert it directly into Section 4 of your final report!

```mermaid
flowchart TD
    %% Define Styles
    classDef data fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:2px;
    classDef db fill:#bfb,stroke:#333,stroke-width:2px;
    classDef ui fill:#fbf,stroke:#333,stroke-width:2px;
    classDef logic fill:#ffb,stroke:#333,stroke-width:2px;
    classDef auto fill:#fca,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;

    %% Data Ingestion Layer
    subgraph Data Ingestion Layer
        Z[Weekly Cron Job] -.->|Triggers Update| A
        A[Cricsheet API] -->|Download ZIP| B(Raw JSON Data)
        B -->|Python Parser| C{NLP Commentary Parsing & Cleaning}
    end
    
    %% Storage Layer
    subgraph Storage Layer
        C -->|Insert Rows| D[(SQLite Database)]
        D -->|1.2 Million Deliveries| E[Historical Match Data]
    end

    %% Processing Layer
    subgraph Processing Layer
        E -->|SQL Queries| F{Pandas Data Aggregation}
        F -->|@st.cache_data| G[Streamlit Memory Cache]
    end

    %% Model/Logic Layer
    subgraph Analytical Engine Layer
        G -->|Partnership Synergies| H[56-Dimensional Tensor]
        G -->|Match Phase Context| H
        G -->|Bowler/Batsman KPIs| H
        H -->|Continuous Momentum Logic| I[Vulnerability Calculation]
    end

    %% Presentation Layer
    subgraph Frontend User Interface
        I -->|JSON / Array| J[Streamlit Dashboard UI]
        J -->|12-Ball User Input Sequence| H
        J -->|Visual Output| K[Next-Ball Predictive Probabilities]
    end

    %% Assign Classes
    class A,B data;
    class C,F process;
    class D,E db;
    class J,K ui;
    class H,I logic;
    class Z auto;
```
