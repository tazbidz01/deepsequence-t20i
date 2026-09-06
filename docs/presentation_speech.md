# DeepSequence T20I: 5-Minute Final Presentation Script

*Instructions: This script matches the strict timeline (1 min Intro, 3.5 min Technical, 45 sec Conclusion). It has been completely rewritten so that Member 1 (Data & ML) only talks about their assigned machine learning and data parsing tasks, and Member 2 (UI & Architecture) only talks about the database, backend routing, and frontend design.*

---

## Member 1 (Time: 0:00 to 2:30)

### Part 1: Project Introduction and Problem Statement (0:00 - 1:00)
**Member 1:** 
"Good afternoon, everyone. Today, we are presenting 'DeepSequence T20I', which is a momentum-aware predictive engine for T20 cricket. 

To state our core problem in a single sentence: traditional cricket analytics rely on static career averages that fail to measure the changing momentum of a live match, causing teams to make multi-million dollar tactical decisions based on flawed data rather than real-time mathematical predictions. Our machine learning project solves this by simulating exact scoreboard pressure to predict the next ball."

### Part 2A: Data Engineering & Machine Learning (1:00 - 2:45)
**Member 1:** 
"My role in this project was Data and Machine Learning. To start, I built a batch parser engine to download and cleanly structure the massive Cricsheet dataset, which gave us over 1.2 million nested JSON match deliveries. 

With the data cleaned, I formulated the multi-parameter input matrices for our PyTorch LSTM model. I designed the engine to read up to a 12-ball rolling sequence and mapped player attributes like specific bowler sub-styles. I also built a custom Natural Language Processing, or NLP, script using regex rules to extract the ball's Line and Length directly from unstructured text commentary. Finally, I connected these NLP features into the sequence predictor tensor, and implemented custom Focal Loss math during training to resolve the severe class imbalances between dot balls and wickets.

I will now pass it to my partner to explain the database architecture and user interface."

---

## Member 2 (Time: 2:45 to 5:00)

### Part 2B: Database Architecture & User Interface (2:45 - 4:15)
**Member 2:** 
"Thank you. My role in this project was UI and Architecture. While my partner built the machine learning models, I had to figure out how to process 1.2 million rows of data instantly without freezing the computer. 

First, I designed the relational database schemas and configured the SQLite storage instances. By using C-level indexing, I established backend utility paths to fetch processed delivery statistics in milliseconds. Next, I designed the frontend dashboard using Streamlit. To prevent the UI from crashing during heavy model predictions, I established strict backend routing pipelines and memory caching to save the ML results directly in the computer's RAM.

I also integrated the interactive grid charts for our Partnership and Bowler Suppression metrics, and connected the frontend visual elements directly to the backend LSTM model endpoints so coaches can test different match commentaries live in the UI."

### Part 3: Cost Analysis & Conclusion (4:15 - 5:00)
**Member 2:** 
"Finally, looking at the conclusion and real-world cost. If a professional T20 franchise used our software, it would be extremely cheap. Our entire technology stack—Python, Pandas, and SQLite—is completely open-source with zero licensing fees. The 1.2-million-row dataset is also completely free. The only cost would be renting an AWS cloud server to host our Streamlit dashboard, which is around $30 a month. 

In conclusion, DeepSequence successfully bridges the gap between basic static stats and real-time pressure, giving coaches a scalable, lightning-fast tactical tool for the modern game."

### The Transition to Live Demo (5:00)
**Member 2:** 
"That concludes our presentation. We will now switch over to our laptop for a quick live demonstration of the dashboard in action."

---

## Advice for the Live Demo (Post-Speech)
*Once the 5 minutes are over and the timer stops, open the Streamlit dashboard.*
1. **Show the Caching Speed:** Point out how instantly the numbers load when you select a player. Mention that it is searching 1.2 million rows instantly because of the RAM cache.
2. **The Partnership Trick:** Set Striker to `RG Sharma`. Set Non-Striker to `KL Rahul` and show the low vulnerability. Then, change the Non-Striker to `SK Raina` and watch the vulnerability spike because of historical scoreboard pressure!
3. **The 12-Ball Sequence:** Change the balls in the dropdown menu (e.g., set three dot balls in a row) and show how the momentum math dynamically changes the Wicket probability progress bar.
