# UI Wireframe & Dashboard Layout Design
**Member 2 Task: Dashboard Layout Wireframing**

This document outlines the visual structure, interactive pages, controls, and grids of the Streamlit dashboard application for DeepSequence-T20I.

---

## 1. Grid Grid Layout (4-Page Structure)

The application utilizes a sidebar controller for universal context filtering and standard top navigation tabs to toggle analytical focus.

```
+-----------------------------------------------------------------------------------+
|  [Header Badge] DeepSequence-T20I: Contextual Batsman Vulnerability Engine       |
+-----------------------------------------------------------------------------------+
|  [Sidebar Controls]         |  [Navigation Tabs]                                  |
|                             |  (Profile) (Commentary Parsing) (Predictor) (PDF)  |
|  * Select Target Batsman    +-----------------------------------------------------+
|    [ Kohli, Virat  v]       |                                                     |
|                             |  [Main Page Container]                              |
|  * Filter Bowler Hand       |                                                     |
|    ( ) Right-Arm            |  (Layout elements, charts, and tables display here)  |
|    ( ) Left-Arm             |                                                     |
|    (*) Both                 |                                                     |
|                             |                                                     |
|  * Filter Bowler Style      |                                                     |
|    [X] Pace   [X] Spin      |                                                     |
+-----------------------------+-----------------------------------------------------+
```

---

## 2. Detailed Page Layouts

### Page 1: Batsman Profile Dashboard
Visualizes static aggregates against contextual splits.
*   **Row 1: KPI Metric Cards**
    *   Total Runs | Total Balls Faced | Strike Rate | Times Dismissed
*   **Row 2: Contextual Matchup Splits (2-Column Grid)**
    *   *Column 1 (Bar Chart):* Strike Rate split by Match Phase (Powerplay vs Middle vs Death).
    *   *Column 2 (Donut Chart):* Dismissals split by Bowler Hand (Right-Arm vs Left-Arm).
*   **Row 3: Bowler Sub-Style Matrix**
    *   A tabular heatmap showing Strike Rate and Average against: Pace, Off-spin, Leg-spin, and Chinaman deliveries.

### Page 2: Commentary NLP Parser Playground
Demonstrates text extraction live.
*   **Input Box:** A large text area where the analyst pastes ball-by-ball commentary (e.g., *"Starc bowls a full delivery outside off-stump, Kohli attempts a drive but edges it to first slip for a dismissal"*).
*   **Extract Button:** Triggers regex parsing logic.
*   **Result Cards (3-Column Grid):**
    *   *Card 1:* Extracted Line (e.g., `outside_off`)
    *   *Card 2:* Extracted Length (e.g., `full`)
    *   *Card 3:* Extracted Shot Intent (e.g., `drive`)

### Page 3: Live Sequence Predictor Simulator
Iteratively simulates sequences of deliveries faced to project next-ball vulnerability.
*   **Sequence Builder Table:** Allows adding 6 consecutive deliveries by picking their variables (Runs, Line, Length, Shot, Bowler style).
*   **Predict Button:** Packages the sequence into a tensor array of shape $(1, 6, 25)$ and runs the LSTM model.
*   **Vulnerability Bar Chart:** Gauges the probability of error/dismissal on the next delivery. Highlights pressure indicators (e.g., 3 consecutive dot balls increases risk).

### Page 4: Strategic Plan-of-Attack PDF Hub
Automated compiler of strategic cards.
*   **Bowler Profile Selection:** Choose target bowler style/hand.
*   **Strategy Summary:** A text description of the weakness sequence (e.g., *"Bowl 2 good-length deliveries outside off-stump, followed by a yorker on middle-leg"*).
*   **Download Button:** Triggers ReportLab to generate `proposal.pdf` containing the strategic cheat sheet.
