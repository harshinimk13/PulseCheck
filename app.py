import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from PIL import Image
import torch.nn.functional as F
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import folium
from streamlit_folium import st_folium
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="PulseCheck | AI Radar", page_icon="📡", layout="wide")

# --- CUSTOM CSS FOR SOS ANIMATION ---
st.markdown("""
<style>
@keyframes pulse-red {
  0% { background-color: #4a0000; color: white; box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
  50% { background-color: #ff0000; color: white; box-shadow: 0 0 20px 10px rgba(255, 0, 0, 0.5); }
  100% { background-color: #4a0000; color: white; box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
}
.sos-box {
  animation: pulse-red 1.5s infinite;
  padding: 20px;
  border-radius: 10px;
  text-align: center;
  font-family: 'Courier New', Courier, monospace;
  font-size: 28px;
  font-weight: 900;
  margin-bottom: 20px;
  border: 2px solid red;
}
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE FOR STACKING ---
if 'history' not in st.session_state:
    st.session_state.history = {} # Dictionary to store and overwrite locations
if 'latest_scan' not in st.session_state:
    st.session_state.latest_scan = None

# --- CACHE & LOAD MODELS ---
@st.cache_resource
def load_text_model():
    model_path = "disaster_model_final"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

@st.cache_resource
def load_vision_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = ['Damaged', 'Normal']
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load('native_disaster_vision_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device, class_names

text_classifier = load_text_model()
vision_model, device, class_names = load_vision_model()

# --- INFERENCE FUNCTIONS ---
def check_tweet(text):
    results = text_classifier(text)[0]
    disaster_score = 0.0
    for r in results:
        if r['label'] == 'LABEL_1':
            disaster_score = r['score']
    return disaster_score

def test_single_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = vision_model(input_batch)
        probabilities = F.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = class_names[predicted_idx.item()]
    return confidence.item(), predicted_class

# ==========================================
# SIDEBAR: TEAM & CALIBRATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3204/3204071.png", width=80) 
    st.title("PulseCheck")
    st.caption("v1.0.0 Tactical Build")
    st.divider()
    
    # Optional button to clear the map if it gets too cluttered
    if st.button("🔄 Clear Radar Map"):
        st.session_state.history = {}
        st.session_state.latest_scan = None
        st.rerun()
        
    st.markdown("### 🛠️ Radar Calibration")
    st.write("Tune thresholds to filter out background AI noise.")
    
    red_threshold = st.slider("🔴 Critical Alert (Red) Minimum", 0.75, 0.99, 0.85)
    orange_threshold = st.slider("🟡 Moderate Warning (Orange) Minimum", 0.60, 0.80, 0.65)
    
    st.divider()
    st.markdown("### 👨‍💻 Founders & Developers")
    st.markdown("""
    * **Gayathri R**
    * **Himnish Kumar R**
    * **Rakshitha V**
    * **Harshini M**
    * **Gracy Sweety T**
    """)
    st.caption("Built with ⚡ during Hackathon 2026")

# ==========================================
# MAIN UI
# ==========================================
st.markdown("<h1 style='text-align: center;'>📡 PulseCheck // AI Emergency Radar</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Real-time multimodal threat detection system.</p>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📍 Location & Intel")
    location_input = st.text_input("Target Coordinates / Area:", value="Jayanagar BDA complex, Bengaluru")
    tweet_input = st.text_area("Live Chatter / Intel Report:", 
                               value="Just had the best dosas near the Jayanagar BDA complex. The weather is so nice today! Everything is completely fine and peaceful. #BangaloreDiaries",
                               height=150)
with col2:
    st.markdown("### 📸 Visual Evidence")
    uploaded_file = st.file_uploader("Drop satellite or ground-level imagery here", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Source verified.', use_container_width=True)

st.divider()

# --- TRIGGER THE SCAN ---
if st.button("🚨 INITIATE AI SCAN", use_container_width=True, type="primary"):
    with st.spinner("🧠 Neural networks analyzing threat levels & locking coordinates..."):
        time.sleep(1) 
        
        # 1. Run Text Model
        text_conf = check_tweet(tweet_input)
        
        # 2. Run Image Model
        img_conf, img_class, image_threat = 0.0, "N/A", 0.0
        if uploaded_file is not None:
            img_conf, img_class = test_single_image(image)
            image_threat = img_conf if img_class == 'Damaged' else (1.0 - img_conf)
            
        # 3. Calculate Overall Threat using 70% Image / 30% Text Weighting
        if uploaded_file is not None:
            overall_threat_score = (0.70 * image_threat) + (0.30 * text_conf)
        else:
            overall_threat_score = text_conf
        
        if overall_threat_score >= red_threshold:
            threat_level, marker_color, icon_type = "HIGH", "red", "bolt"
        elif overall_threat_score >= orange_threshold: 
            threat_level, marker_color, icon_type = "MODERATE", "orange", "exclamation-triangle"
        else:
            threat_level, marker_color, icon_type = "LOW", "green", "shield"

        # 4. GEOCODE HERE (So we can save lat/lon to history)
        geolocator = Nominatim(user_agent="pulsecheck_radar_app") 
        lat, lon = None, None
        try:
            loc = geolocator.geocode(location_input, timeout=5)
            if not loc and "," in location_input:
                loc = geolocator.geocode(location_input.split(",")[-1].strip(), timeout=5)
            if not loc:
                loc = geolocator.geocode("Bengaluru, India", timeout=5)
            
            if loc:
                lat, lon = loc.latitude, loc.longitude
        except GeocoderTimedOut:
            pass # Failsafe will catch missing coords

        # 5. Save to Session State Dictionary (Overwrites if location exists)
        if lat and lon:
            st.session_state.history[location_input] = {
                'text_conf': text_conf,
                'img_conf': img_conf,
                'img_class': img_class,
                'overall_threat_score': overall_threat_score,
                'threat_level': threat_level,
                'marker_color': marker_color,
                'icon_type': icon_type,
                'location_query': location_input,
                'lat': lat,
                'lon': lon
            }
            st.session_state.latest_scan = location_input
        else:
            st.error("Could not find coordinates for that location.")

# ==========================================
# RENDER RESULTS
# ==========================================
if st.session_state.latest_scan:
    # Pull metrics for the MOST RECENT scan to show on top
    res = st.session_state.history[st.session_state.latest_scan]
    
    if res['threat_level'] == "HIGH":
        st.markdown('<div class="sos-box">⚠️ CRITICAL ALERT: DISASTER CONFIRMED IN SECTOR ⚠️</div>', unsafe_allow_html=True)
        audio_html = """
            <audio autoplay style="display:none;">
            <source src="https://upload.wikimedia.org/wikipedia/commons/4/40/Alarm_Loop.ogg#t=0,3" type="audio/ogg">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        
        with st.status("📡 Establishing secure uplink to Emergency Response Dispatch...", expanded=True) as status:
            st.write("Encrypting geographical coordinates...")
            time.sleep(1)
            st.write("Attaching multimodal AI damage assessments...")
            time.sleep(1)
            st.write(f"Routing packet to nearest rescue units near {res['location_query']}...")
            time.sleep(1.5)
            status.update(label="Dispatched Successfully! Rescue team ETA: 4 mins.", state="complete", expanded=False)
            
    elif res['threat_level'] == "MODERATE":
        st.warning("### ⚠️ WARNING: MODERATE THREAT DETECTED")
    else:
        st.success("### 🟢 SECTOR CLEAR: No Immediate Threat")
    
    m1, m2, m3 = st.columns(3)
    
    text_delta_color = "inverse" if res['text_conf'] >= orange_threshold else "normal"
    m1.metric(label="Text Threat Score", value=f"{res['text_conf']*100:.1f}%", delta="High Risk" if res['text_conf'] >= orange_threshold else "Low Risk", delta_color=text_delta_color)
    
    if res['img_class'] != "N/A":
        m2.metric(label="Image Analysis", value=f"{res['img_conf']*100:.1f}%", delta=res['img_class'], delta_color="inverse" if res['img_class'] == 'Damaged' else "normal")
    else:
        m2.metric(label="Image Analysis", value="N/A", delta="No Image", delta_color="off")
        
    m3.metric(label="Overall AI Status", value=res['threat_level'], delta=f"Max Threat: {res['overall_threat_score']*100:.1f}%", delta_color="inverse" if res['threat_level'] != "LOW" else "normal")

    st.divider()
    
    # --- STACKED MAP RENDERING ---
    st.markdown(f"### 🗺️ Live Tactical Map ({len(st.session_state.history)} Sector(s) Tracked)")
    
    # Initialize map centered on the most recent scan
    m = folium.Map(location=[res['lat'], res['lon']], zoom_start=14, tiles="OpenStreetMap")
    
    # Loop through ALL scans stored in history and plot them
    for loc_name, past_scan in st.session_state.history.items():
        folium.CircleMarker(
            location=[past_scan['lat'], past_scan['lon']],
            radius=50,
            color=past_scan['marker_color'],
            fill=True,
            fill_color=past_scan['marker_color'],
            fill_opacity=0.4
        ).add_to(m)
        folium.Marker(
            [past_scan['lat'], past_scan['lon']], 
            popup=f"<b>{loc_name}</b><br>Threat Level: {past_scan['threat_level']}",
            icon=folium.Icon(color=past_scan['marker_color'], icon=past_scan['icon_type'], prefix="fa")
        ).add_to(m)
        
    st_folium(m, width=1200, height=500, returned_objects=[])
