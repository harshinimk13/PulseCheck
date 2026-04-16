# 📡 PulseCheck — AI Emergency Radar

**PulseCheck** is a real-time, multimodal disaster detection system built for rapid situational awareness. It combines a fine-tuned NLP model and a computer vision model to analyse social media chatter and on-ground imagery simultaneously, then plots threat levels on a live interactive map.

> Built with ⚡ during **Hackanova 2026** by Team PulseCheck.

---

## ✨ Features

- **Dual-modal threat analysis** — fuses text (social media / intel reports) and image (satellite / ground-level photos) signals into a single threat score
- **DistilBERT text classifier** — fine-tuned on disaster-related text for fast, accurate tweet/report classification
- **ResNet-18 vision model** — classifies uploaded images as `Damaged` or `Normal`
- **Live geocoding** — resolves any location string to coordinates using Nominatim
- **Interactive radar map** — renders colour-coded threat markers with Folium; stacks multiple scans in one session
- **Adjustable alert thresholds** — sidebar sliders let operators calibrate sensitivity on the fly
- **Animated SOS alert** — pulsing red banner + audio alarm triggered on critical detections
- **Auto-dispatch simulation** — simulates routing to emergency response units on HIGH threat

---

## 🗂️ Project Structure

```
PulseCheck/
├── app.py                              # Streamlit application (main entry point)
├── requirements.txt                    # Python dependencies
├── native_disaster_vision_model.pth    # Fine-tuned ResNet-18 weights
└── disaster_model_final/              # Fine-tuned DistilBERT model
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── training_args.bin
    └── vocab.txt
```

---

## 🧠 How It Works

### 1. Text Analysis (NLP)
Social media posts or intel reports are passed through a **DistilBERT** model (`distilbert-base-uncased`) fine-tuned for binary sequence classification:
- `LABEL_0` — Non-disaster
- `LABEL_1` — Disaster

The confidence score for `LABEL_1` becomes the **text threat score**.

### 2. Image Analysis (Computer Vision)
Uploaded images are pre-processed and passed through a fine-tuned **ResNet-18** model:
- Output: `Damaged` or `Normal` with a confidence score
- If classified as `Damaged`, the confidence directly contributes to the threat score; if `Normal`, the inverse is used.

### 3. Combined Threat Score
```
Overall Threat = (0.70 × Image Threat) + (0.30 × Text Threat)   # when image is provided
Overall Threat = Text Threat                                       # text-only mode
```

### 4. Threat Classification

| Level | Score Range (default) | Map Marker | Action |
|---|---|---|---|
| 🟢 LOW | < 0.65 | Green | Sector clear |
| 🟡 MODERATE | 0.65 – 0.84 | Orange | Warning issued |
| 🔴 HIGH | ≥ 0.85 | Red | SOS + dispatch simulation |

> Thresholds are adjustable via the sidebar sliders.

### 5. Geocoding & Mapping
The location string is resolved to lat/lon using **Geopy (Nominatim)**. All scans in a session are plotted as stacked markers on a **Folium** map with popups showing threat details. The map re-centres on the most recent scan.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone / unzip the project
cd PulseCheck

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).

> **Note:** Both model files (`disaster_model_final/` and `native_disaster_vision_model.pth`) must be present in the project root for the app to start.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `torch` + `torchvision` | Vision model inference (CPU build) |
| `transformers` | DistilBERT text classification pipeline |
| `Pillow` | Image loading and pre-processing |
| `geopy` | Location string → lat/lon geocoding |
| `folium` | Interactive map rendering |
| `streamlit-folium` | Embeds Folium maps inside Streamlit |

---

## 🖥️ Usage

1. Enter a **location** (e.g., `Jayanagar BDA complex, Bengaluru`)
2. Paste a **tweet / intel report** into the text area
3. Optionally upload a **satellite or ground-level image** (JPG/PNG)
4. Click **🚨 INITIATE AI SCAN**
5. Review the threat score metrics and the live radar map
6. Run multiple scans — each location is tracked and stacked on the map
7. Use **🔄 Clear Radar Map** in the sidebar to reset the session

---

## 👨‍💻 Team

| Name |
|---|
| Gayathri R |
| Himnish Kumar R |
| Rakshitha V |
| Harshini M |
| Gracy Sweety T |

---

## ⚠️ Limitations & Notes

- Geocoding relies on the **Nominatim** public API — results may time out for obscure or misspelled locations; the app falls back to Bengaluru, India if resolution fails.
- Models are loaded in **CPU mode** by default; a CUDA GPU will be used automatically if available.
- The emergency dispatch sequence is a **simulation only** — no real alerts are sent.
- The `streamlit-folium` version in `requirements.txt` has no pinned version number; pin it (e.g., `streamlit-folium==0.18.0`) for reproducible installs.
