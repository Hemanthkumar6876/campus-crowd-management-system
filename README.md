# Campus Crowd Density & Congestion Prediction System

Computer-vision + machine-learning prototype with a dark, glass-style Streamlit dashboard.

| Module | Implementation |
|---|---|
| Video/Image processing, person detection | `detector.py` - YOLOv8 (ultralytics) with OpenCV HOG fallback |
| Crowd counting, density estimation | counts, occupancy % and people per 100 m² per location |
| Congestion classification | Low / Medium / High / Critical, adjustable thresholds (`config.py`) |
| Historical pattern analysis | SQLite history (`database.py`), hour-of-day heatmaps, chronic hotspot ranking |
| Crowd prediction (15-30 min) | `predictor.py` - statistical model + scikit-learn Gradient Boosting + ensemble |
| Hotspot detection, heatmaps | image-level hotspots + campus-level density map (`heatmap.py`) |
| Alert generation | threshold alerts, forecast alerts, acknowledgeable log |
| Monitoring dashboard | `app.py` + `style.py` |
| Evaluation & reports | walk-forward back-test vs persistence baseline; PDF/CSV export (`reports.py`) |

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
Open http://localhost:8501. First launch creates 14 days of simulated history.
Optional, for accurate counting in dense crowds: `pip install ultralytics`.

## Deploy
**Streamlit Community Cloud (free, gives a public link)**
1. Push this folder to a GitHub repo (`git init`, `git add .`, `git commit`, `git push`).
2. Go to share.streamlit.io, sign in with GitHub, choose the repo, main file `app.py`, click Deploy.
(For YOLO on Cloud add `ultralytics` to requirements.txt.)

**Docker**
    docker build -t campus-crowd .
    docker run -p 8501:8501 campus-crowd

## Real data
Upload photos/videos on "Upload & Detect" and save them, or import a CSV (`timestamp,location,count`)
on "Historical Database". Edit `LOCATIONS` in `config.py` to match your campus.
