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
| Monitoring dashboard, login | `app.py`, `style.py`, `auth.py` |
| Evaluation & reports | walk-forward back-test vs persistence baseline; PDF/CSV export (`reports.py`) |

## Login
The app opens on a sign-in page. First launch creates two accounts:
`admin / admin123` (full access, User Management page, can clear data) and `staff / staff123` (view only).
New people can use "Request access" to create a viewer account. Passwords are stored as salted PBKDF2 hashes.
For deployment set `CAMPUS_ADMIN_USER`, `CAMPUS_ADMIN_PASSWORD` (before the first run) and `SHOW_DEMO_CREDS=0`.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
Open http://localhost:8501. First launch creates 14 days of simulated history.
Optional, for accurate counting in dense crowds: `pip install ultralytics`.

## Deploy
**Streamlit Community Cloud (free, gives a public link)**
1. Push this folder to a GitHub repo (keep `requirements.txt`, `packages.txt` and `yolov8n.pt` in the repo root).
2. Go to share.streamlit.io, sign in with GitHub, choose the repo, main file `app.py`.
   Under *Advanced settings* choose Python 3.11, then Deploy.
3. `requirements.txt` installs YOLOv8 with CPU-only PyTorch for accurate counting. If the free tier runs out of memory,
   delete the last 4 lines: the app then falls back to the weaker OpenCV HOG detector.
4. Set a real admin password and hide the demo accounts (Settings > Secrets):
   (Streamlit Cloud exposes top-level secrets as environment variables, set them before the first run):
       CAMPUS_ADMIN_PASSWORD = "choose-a-strong-one"
       SHOW_DEMO_CREDS = "0"

**Live Camera when deployed**
A deployed server has no webcam, so use *Live Camera > Browser camera*: your browser opens the camera
(allow access when asked, the site must be https, which Streamlit Cloud is) and streams it to the server for detection.
Tuning: *Speed / quality* (Fast = least lag), *YOLO model* (yolov8s is more accurate), *Dense crowd mode* (finds small /
far people, about 5x slower), *Detection confidence* and *Count correction factor*.
The "Webcam on this computer" and IP/RTSP options only work when the app runs on the machine that owns the camera.
If the video never starts on a school/office/mobile network, add a TURN relay server in *Settings > Secrets*:
    [turn]
    urls = "turn:YOUR_TURN_HOST:3478"
    username = "USER"
    credential = "PASSWORD"
(free tiers exist at providers such as Metered or Twilio). In the secrets file keep the top-level lines (CAMPUS_ADMIN_PASSWORD...) above the [turn] block.

Note: Streamlit Cloud storage is temporary. The SQLite database (users, history, alerts) resets when the app restarts.
For permanent data use a server/VM or Docker with a mounted `data/` volume.

**Docker**
    docker build -t campus-crowd .
    docker run -p 8501:8501 -v campus_data:/app/data campus-crowd

## Real data
Upload photos/videos on "Upload & Detect" and save them, or import a CSV (`timestamp,location,count`)
on "Historical Database". Edit `LOCATIONS` in `config.py` to match your campus.
