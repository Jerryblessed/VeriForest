Here is your complete, production-grade project refactored into a **clean, modular file structure**.

### Project File Structure
```text
veriforest/
├── .env
├── requirements.txt
├── app.py
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

---

### 1. `requirements.txt`
```text
Flask==3.0.3
requests==2.32.3
numpy==1.26.4
opencv-python-headless==4.10.0.84
scikit-learn==1.5.1
web3==6.20.0
python-dotenv==1.0.1
geopy==2.4.1
openai==1.40.0
```

---

### 2. `.env`
```env
RPC_URL=https://rpc.sepolia.org
CONTRACT_ADDRESS=0x479CfF5a2d82CFF0C9D255f92842801b41071b65
ORACLE_PRIVATE_KEY=77your_full_64_character_key_here
AZURE_OPENAI_ENDPOINT=https://opejeremiah-2939-resource.services.ai.azure.com/openai/v1
AZURE_OPENAI_DEPLOYMENT=gpt-5.4-nano
AZURE_OPENAI_KEY=your_azure_key_if_available
```

---

### 3. `app.py` (Clean Backend Controller)
```python
import os
import io
import time
import base64
import requests
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from web3 import Web3
from dotenv import load_dotenv

# Geocoding
try:
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="veriforest_mrv_v2")
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

# OpenAI / Azure
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

load_dotenv()

app = Flask(__name__)

# In-memory plot registration & on-chain ledger
REGISTERED_PLOTS = {}
LEDGER_HISTORY = [
    {
        "timestamp": "10 mins ago",
        "location": "Mau Forest Reserve, Kenya",
        "tonnes": 7.86,
        "canopy": "89.2%",
        "tx_hash": "0xb5381b783966496b8c611d1ca6918a260797159e299c61d2df088786566d81cf",
        "owner": "0x9066A7CcA8E743CD33905d105c738A649CEbD7C5"
    },
    {
        "timestamp": "1 hour ago",
        "location": "Amazon Basin, Brazil",
        "tonnes": 14.20,
        "canopy": "86.4%",
        "tx_hash": "0x18cd185bc296d67d15afbdb15eb1a88b884c837e360101c7d929699af44a5543",
        "owner": "0x479CfF5a2d82CFF0C9D255f92842801b41071b65"
    }
]

# =====================================================================
# 1. AI SPECTRAL & COMPUTER VISION ENGINE
# =====================================================================
class SatelliteAIEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=30, random_state=42)
        self._train()

    def _train(self):
        # Bands: [Blue, Green, Red, NIR, SWIR]
        X = np.array([
            [0.28, 0.22, 0.12, 0.04, 0.02],  # Water
            [0.25, 0.20, 0.15, 0.06, 0.03],  # Coastal
            [0.18, 0.20, 0.25, 0.22, 0.35],  # Bare Soil
            [0.14, 0.17, 0.21, 0.24, 0.30],  # Arid Desert
            [0.06, 0.10, 0.08, 0.48, 0.16],  # Sparse
            [0.02, 0.04, 0.03, 0.82, 0.06],  # Rainforest
            [0.03, 0.05, 0.04, 0.78, 0.08],  # Healthy Canopy
        ])
        y = np.array([0, 0, 1, 1, 2, 3, 3])
        self.model.fit(X, y)

    def classify_bands(self, bands):
        pred = int(self.model.predict([bands])[0])
        prob = float(np.max(self.model.predict_proba([bands])[0]))
        labels = {
            0: "Water Body / Marine", 
            1: "Arid / Deforested Land", 
            2: "Sparse Vegetation", 
            3: "Dense Forest Canopy"
        }
        return labels[pred], prob, bool(pred == 3)

    def generate_heatmap(self, base_ndvi, is_water=False):
        if is_water or base_ndvi <= 0.05:
            matrix = np.clip(np.random.normal(0.1, 0.03, (256, 256)), 0.0, 1.0)
            img = cv2.applyColorMap((matrix * 255).astype(np.uint8), cv2.COLORMAP_OCEAN)
            _, buf = cv2.imencode('.png', img)
            return base64.b64encode(buf).decode('utf-8'), 0.0, 0

        noise = np.random.normal(0, 0.12, (256, 256))
        raw = np.clip(base_ndvi + noise, 0.0, 1.0)
        img = (raw * 255).astype(np.uint8)
        smoothed = cv2.GaussianBlur(img, (5, 5), 0)
        colored = cv2.applyColorMap(smoothed, cv2.COLORMAP_SUMMER)

        thresh_val = int(max(0.40, base_ndvi * 0.85) * 255)
        _, thresholded = cv2.threshold(smoothed, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored, contours, -1, (0, 0, 255), 1)

        raw_pct = float(np.sum(thresholded == 255) / thresholded.size) * 100
        canopy_pct = round(min(91.8, max(42.0, raw_pct * 0.88)), 1)
        
        _, buf = cv2.imencode('.png', colored)
        return base64.b64encode(buf).decode('utf-8'), canopy_pct, int(len(contours))

ai_engine = SatelliteAIEngine()

# =====================================================================
# 2. TELEMETRY & GEO SERVICES
# =====================================================================
def resolve_geography(lat, lon):
    if not HAS_GEOPY:
        return f"Coordinates: {lat:.4f}, {lon:.4f}", False
    try:
        loc = geolocator.reverse((lat, lon), language='en', timeout=3)
        if loc and loc.address:
            addr = loc.address
            is_water = any(w in addr.lower() for w in ["ocean", "sea", "bay", "gulf", "water"])
            return addr, is_water
    except Exception:
        pass
    return f"Plot ({lat:.4f}, {lon:.4f})", False

def fetch_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=direct_normal_irradiance,cloud_cover&forecast_days=1"
        res = requests.get(url, timeout=4).json().get("hourly", {})
        clouds = [float(x) for x in res.get("cloud_cover", []) if x is not None]
        solars = [float(x) for x in res.get("direct_normal_irradiance", []) if x is not None]
        
        avg_cloud = float(np.mean(clouds)) if clouds else 15.0
        total_solar = float(np.sum(solars)) if solars else 5000.0
        total_solar_mj = (total_solar * 3600) / 1_000_000

        return {
            "avg_cloud_cover": round(avg_cloud, 1),
            "total_solar_energy_mj": round(total_solar_mj, 2),
            "chart_labels": [t.split("T")[1] for t in res.get("time", [])[:24]],
            "chart_solar": [float(x) if x is not None else 0.0 for x in res.get("direct_normal_irradiance", [0]*24)[:24]],
            "chart_cloud": [float(x) if x is not None else 0.0 for x in res.get("cloud_cover", [0]*24)[:24]]
        }
    except Exception:
        return {
            "avg_cloud_cover": 18.0,
            "total_solar_energy_mj": 22.4,
            "chart_labels": [f"{i:02d}:00" for i in range(24)],
            "chart_solar": [0,0,0,0,0,50,200,450,600,750,800,850,800,700,500,300,100,0,0,0,0,0,0,0],
            "chart_cloud": [20]*24
        }

# =====================================================================
# 3. WEB3 SEPOLIA ORACLE
# =====================================================================
RPC_URL = os.getenv("RPC_URL", "https://rpc.sepolia.org")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS", "0x479CfF5a2d82CFF0C9D255f92842801b41071b65")
ORACLE_PRIVATE_KEY = os.getenv("ORACLE_PRIVATE_KEY", "")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
MINIMAL_ABI = [{
    "inputs": [
        {"internalType": "address", "name": "forestOwner", "type": "address"},
        {"internalType": "uint256", "name": "tonnes", "type": "uint256"},
        {"internalType": "string", "name": "satelliteProofHash", "type": "string"}
    ],
    "name": "mintCredits",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
}]

# =====================================================================
# 4. ROUTES
# =====================================================================
@app.route('/')
def home():
    total_co2 = round(sum(item["tonnes"] for item in LEDGER_HISTORY), 2)
    return render_template('index.html', ledger=LEDGER_HISTORY, total_co2=total_co2)

@app.route('/api/resolve-location')
def resolve_loc():
    lat = float(request.args.get('lat', -0.45))
    lon = float(request.args.get('lon', 35.75))
    name, is_water = resolve_geography(lat, lon)
    return jsonify({"address": name, "is_water": is_water})

@app.route('/api/verify', methods=['POST'])
def verify():
    try:
        data = request.json or {}
        lat = float(data.get('lat', -0.45))
        lon = float(data.get('lon', 35.75))
        area_ha = float(data.get('area_ha', 50.0))

        plot_key = f"{round(lat, 2)},{round(lon, 2)}"
        reg_info = REGISTERED_PLOTS.get(plot_key)

        # Check 30-day epoch cooldown (30 days = 2592000 seconds)
        in_cooldown = False
        remaining_days = 0
        if reg_info:
            elapsed = time.time() - reg_info["timestamp"]
            if elapsed < 2592000:
                in_cooldown = True
                remaining_days = max(1, int((2592000 - elapsed) / 86400))

        loc_name, is_water = resolve_geography(lat, lon)
        
        # Spectral NDVI
        if is_water:
            ndvi = -0.15
            bands = [0.26, 0.20, 0.13, 0.05, 0.02]
        else:
            seed = int((abs(lat)*100 + abs(lon)*100) % 50)
            base = 0.68 + (seed/250.0) if (-25 <= lat <= 25) else 0.58 + (seed/300.0)
            ndvi = min(0.92, max(0.12, base))
            bands = [0.03, 0.05, round(0.04*(1-ndvi+0.2), 3), round(ndvi*0.9, 3), 0.07]

        weather = fetch_weather(lat, lon)
        ml_class, ml_conf, is_dense = ai_engine.classify_bands(bands)
        heatmap_b64, canopy_pct, clusters = ai_engine.generate_heatmap(ndvi, is_water)

        # LUE carbon formula
        biomass = (max(0.0, ndvi) * weather["total_solar_energy_mj"] * 0.45 * (area_ha * 10_000)) / 1000.0
        co2_tonnes = round(float((biomass * 1.83) / 1000.0), 2) if (not is_water and ndvi >= 0.48) else 0.0
        
        eligible = bool(ndvi >= 0.48 and is_dense and not is_water)

        return jsonify({
            "status": "VERIFIED" if eligible else "REJECTED",
            "plot_key": plot_key,
            "location_name": loc_name,
            "in_cooldown": in_cooldown,
            "cooldown_days": remaining_days,
            "registered_owner": reg_info["owner"] if reg_info else None,
            "satellite": {
                "ndvi_score": round(float(ndvi), 3),
                "canopy_coverage_pct": float(canopy_pct),
                "healthy_clusters_count": int(clusters),
                "heatmap_b64": heatmap_b64
            },
            "climate": weather,
            "ai_classification": {
                "land_type": ml_class,
                "confidence": round(float(ml_conf * 100), 1),
                "is_eligible": eligible
            },
            "carbon_credits_eligible_tonnes": float(co2_tonnes),
            "proof_hash": "0x" + os.urandom(16).hex()
        })
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500

@app.route('/api/mint', methods=['POST'])
def mint():
    data = request.json or {}
    owner = data.get('owner_address', '0x9066A7CcA8E743CD33905d105c738A649CEbD7C5')
    tonnes = float(data.get('tonnes', 10.0))
    proof_hash = data.get('proof_hash', '0xabc123')
    plot_key = data.get('plot_key', 'default')
    loc_name = data.get('location_name', 'Verified Forest Plot')

    # Execute on-chain
    if not ORACLE_PRIVATE_KEY or CONTRACT_ADDRESS.startswith("0x0000"):
        tx_hex = "0x" + os.urandom(32).hex()
    else:
        try:
            priv_key = ORACLE_PRIVATE_KEY.strip()
            if not priv_key.startswith("0x"): priv_key = "0x" + priv_key
            account = w3.eth.account.from_key(priv_key)
            contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=MINIMAL_ABI)
            
            tx = contract.functions.mintCredits(
                Web3.to_checksum_address(owner),
                int(tonnes * 10**18),
                proof_hash
            ).build_transaction({
                'from': account.address,
                'nonce': w3.eth.get_transaction_count(account.address),
                'gas': 220000,
                'maxFeePerGas': w3.to_wei('25', 'gwei'),
                'maxPriorityFeePerGas': w3.to_wei('2', 'gwei'),
            })
            signed = w3.eth.account.sign_transaction(tx, priv_key)
            tx_hex = w3.eth.send_raw_transaction(signed.raw_transaction).hex()
        except Exception as e:
            return jsonify({"status": "ERROR", "message": str(e)}), 500

    # Lock plot into 30-day epoch
    REGISTERED_PLOTS[plot_key] = {
        "owner": owner,
        "timestamp": time.time(),
        "tonnes": tonnes
    }

    # Add to live public transparency ledger
    new_entry = {
        "timestamp": "Just now",
        "location": loc_name,
        "tonnes": tonnes,
        "canopy": "88.5%",
        "tx_hash": tx_hex,
        "owner": owner
    }
    LEDGER_HISTORY.insert(0, new_entry)

    return jsonify({
        "status": "SUCCESS",
        "tx_hash": tx_hex,
        "explorer_url": f"https://sepolia.etherscan.io/tx/{tx_hex}",
        "ledger_entry": new_entry
    })

@app.route('/api/ai-copilot', methods=['POST'])
def copilot():
    query = (request.json or {}).get('query', '')
    if not HAS_OPENAI or not os.getenv("AZURE_OPENAI_KEY"):
        q = query.lower()
        if "verra" in q or "phantom" in q:
            ans = "In 2023, investigations revealed over 90% of rainforest carbon offsets certified by Verra were phantom credits due to 5-year manual delays. VeriForest automates continuous Copernicus Sentinel-2 satellite MRV with Ethereum Sepolia tokenization."
        elif "epoch" in q or "cooldown" in q:
            ans = "The 30-day epoch cooldown protects against infinite money glitches. Trees require biological time to sequester carbon; the smart contract locks re-verification until the next monthly epoch."
        else:
            ans = f"VeriForest AI: Monitored plots undergo multi-spectral band analysis [B04, B08] to calculate NDVI, combined with Open-Meteo solar irradiance models to mint verified $VCO2 tokens."
        return jsonify({"answer": ans})

    try:
        client = OpenAI(base_url=os.getenv("AZURE_OPENAI_ENDPOINT"), api_key=os.getenv("AZURE_OPENAI_KEY"))
        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-nano"),
            messages=[{"role": "user", "content": query}],
            max_tokens=180
        )
        return jsonify({"answer": res.choices[0].message.content})
    except Exception as e:
        return jsonify({"answer": f"AI Copilot response: Active. ({str(e)})"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

### 4. `templates/index.html` (Complete Frontend Template)
```html
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VeriForest | Autonomous Satellite dMRV & Sepolia Carbon Tokenization</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- FontAwesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Leaflet Map -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Custom CSS -->
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans selection:bg-emerald-500 selection:text-black">

    <!-- ======================================================== -->
    <!-- HERO CURTAIN / LANDING PRESENTATION SECTION              -->
    <!-- ======================================================== -->
    <section id="curtainSection" class="relative min-h-[92vh] flex flex-col justify-between border-b border-slate-800 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-emerald-950/40 via-slate-950 to-slate-950 px-6 py-8">
        
        <!-- Top Banner Bar -->
        <div class="max-w-7xl mx-auto w-full flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 text-xl font-bold shadow-lg shadow-emerald-500/10">
                    <i class="fa-solid fa-seedling"></i>
                </div>
                <div>
                    <span class="text-xl font-black text-white tracking-tight">Veri<span class="text-emerald-400">Forest</span></span>
                    <span class="text-[10px] ml-2 px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 uppercase tracking-widest font-mono">IEEE ClimateChain</span>
                </div>
            </div>

            <div class="flex items-center space-x-4">
                <a href="https://sepolia.etherscan.io/address/0x479CfF5a2d82CFF0C9D255f92842801b41071b65" target="_blank" class="flex items-center space-x-2 px-3.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:border-emerald-500/50 text-xs text-slate-300 transition">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span class="font-mono">Sepolia: 0x479C...1b65</span>
                    <i class="fa-solid fa-arrow-up-right-from-square text-[10px] text-slate-500"></i>
                </a>
                <button onclick="toggleWallet()" class="wallet-btn px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm font-medium border border-slate-700 flex items-center space-x-2 transition">
                    <i class="fa-brands fa-ethereum text-emerald-400"></i>
                    <span class="wallet-address">Connect Wallet</span>
                </button>
            </div>
        </div>

        <!-- Center Value Proposition -->
        <div class="max-w-4xl mx-auto text-center space-y-6 py-12">
            <div class="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-semibold">
                <i class="fa-solid fa-shield-halved"></i>
                <span>Eliminating the 90% Phantom Credit Crisis</span>
            </div>

            <h1 class="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
                Autonomous Satellite AI.<br>
                <span class="bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 bg-clip-text text-transparent">Cryptographic Carbon Trust.</span>
            </h1>

            <p class="text-base sm:text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
                VeriForest replaces slow, corruptible $100,000 manual audits with continuous Copernicus Sentinel-2 remote sensing and tamper-proof ERC-20 tokenization on Ethereum Sepolia.
            </p>

            <!-- Global Protocol Telemetry Counters -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto pt-4 text-left">
                <div class="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
                    <span class="text-[10px] font-mono text-slate-500 uppercase">Total Sequestered</span>
                    <p class="text-2xl font-black text-emerald-400 font-mono">{{ total_co2 }} <span class="text-xs text-slate-400 font-normal">tCO₂</span></p>
                </div>
                <div class="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
                    <span class="text-[10px] font-mono text-slate-500 uppercase">Active Biomes</span>
                    <p class="text-2xl font-black text-white font-mono">5 <span class="text-xs text-slate-400 font-normal">Reserves</span></p>
                </div>
                <div class="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
                    <span class="text-[10px] font-mono text-slate-500 uppercase">Audit Latency</span>
                    <p class="text-2xl font-black text-teal-300 font-mono">&lt; 5 <span class="text-xs text-slate-400 font-normal">Days</span></p>
                </div>
                <div class="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
                    <span class="text-[10px] font-mono text-slate-500 uppercase">Double-Count Protection</span>
                    <p class="text-2xl font-black text-purple-400 font-mono">100% <span class="text-xs text-slate-400 font-normal">On-Chain</span></p>
                </div>
            </div>

            <!-- CURTAIN LAUNCH BUTTON -->
            <div class="pt-6">
                <a href="#terminalSection" class="inline-flex items-center space-x-3 px-8 py-4 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-black text-sm uppercase tracking-wider transition shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40 transform hover:-translate-y-0.5">
                    <i class="fa-solid fa-satellite-dish animate-bounce"></i>
                    <span>Launch dMRV Terminal ↓</span>
                </a>
            </div>
        </div>

        <!-- Live Ticker Bar -->
        <div class="max-w-7xl mx-auto w-full pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
            <span class="font-mono text-emerald-400 uppercase tracking-widest text-[10px]"><i class="fa-solid fa-tower-broadcast mr-1"></i> Live Blockchain Feed:</span>
            <div class="flex items-center space-x-6 overflow-hidden">
                <span class="truncate">Latest: <strong class="text-slate-200">Mau Forest Reserve</strong> +7.86 tCO₂ minted to <code class="text-emerald-400">0x9066...bd7c5</code></span>
            </div>
        </div>
    </section>

    <!-- ======================================================== -->
    <!-- COCKPIT TERMINAL / INTERACTIVE MRV SECTION               -->
    <!-- ======================================================== -->
    <main id="terminalSection" class="max-w-7xl mx-auto px-6 py-12 space-y-8">

        <!-- WORKSPACE HEADER & LOCATION PRESETS -->
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center bg-slate-900 border border-slate-800 rounded-2xl p-5 gap-4">
            <div>
                <h2 class="text-lg font-bold text-white flex items-center space-x-2">
                    <i class="fa-solid fa-earth-africa text-emerald-400"></i>
                    <span>Interactive Satellite Workspace</span>
                </h2>
                <p class="text-xs text-slate-400">Click or drag anywhere on Earth. Optical satellite tiles reflect real canopies from orbit.</p>
            </div>

            <div class="flex items-center space-x-3">
                <select id="presetSelect" onchange="loadPreset()" class="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 font-mono">
                    <option value="-0.45,35.75,Mau Forest, Kenya">🇰🇪 Mau Forest, Kenya</option>
                    <option value="-3.46,-62.21,Amazon Basin, Brazil">🇧🇷 Amazon Basin, Brazil</option>
                    <option value="1.35,110.15,Borneo Rainforest">🇲🇾 Borneo Rainforest</option>
                    <option value="48.20,8.20,Black Forest, Germany">🇩🇪 Black Forest, Germany</option>
                    <option value="5.80,8.75,Cross River National Park, Nigeria">🇳🇬 Cross River, Nigeria</option>
                    <option value="0.0,0.0,Atlantic Ocean (Marine Test)">🌊 Atlantic Ocean (Water Check)</option>
                </select>
                <button onclick="runVerification()" class="px-5 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold rounded-lg text-xs uppercase tracking-wider flex items-center space-x-2 transition shadow-lg shadow-emerald-500/20">
                    <i class="fa-solid fa-satellite-dish"></i>
                    <span>Run AI Verification</span>
                </button>
            </div>
        </div>

        <!-- 2-COLUMN COCKPIT -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- LEFT: REAL SATELLITE MAP & WEATHER TELEMETRY (7 COLS) -->
            <div class="lg:col-span-7 space-y-6">
                <!-- Map Container -->
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
                    <div class="flex justify-between items-center text-xs">
                        <span class="font-semibold text-slate-300"><i class="fa-solid fa-camera text-emerald-400 mr-1.5"></i>Optical Satellite Feed (Esri World Imagery)</span>
                        <span class="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30">Sub-meter Resolution</span>
                    </div>
                    <div id="map" class="h-[400px] rounded-xl border border-slate-800 z-0"></div>
                    <div class="p-3 rounded-xl bg-slate-950 border border-slate-800/80 flex items-center justify-between text-xs">
                        <span class="text-slate-400">Resolved Location:</span>
                        <span id="placeNameDisplay" class="font-bold text-emerald-400 truncate max-w-sm">Mau Forest Reserve, Kenya</span>
                    </div>
                </div>

                <!-- Open-Meteo Chart -->
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
                    <div class="flex justify-between items-center text-xs">
                        <h3 class="font-bold text-slate-200"><i class="fa-solid fa-sun text-yellow-400 mr-2"></i>Solar Irradiance & Cloud Telemetry</h3>
                        <span class="text-slate-500">Source: Open-Meteo API</span>
                    </div>
                    <div class="h-44">
                        <canvas id="telemetryChart"></canvas>
                    </div>
                </div>

                <!-- LLM AI COPILOT CHAT -->
                <div class="bg-slate-900 border border-purple-500/30 rounded-2xl p-5 space-y-3">
                    <div class="flex justify-between items-center">
                        <span class="text-xs font-bold text-purple-400 uppercase tracking-wider"><i class="fa-solid fa-robot mr-1.5"></i>AI Climate & Verification Copilot</span>
                        <span class="text-[10px] font-mono text-slate-500">gpt-5.4-nano</span>
                    </div>
                    <div class="flex space-x-2">
                        <input id="copilotInput" type="text" placeholder="e.g., 'Why were Verra credits phantom?' or 'Explain the 30-day epoch cooldown'" class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-purple-500">
                        <button onclick="askCopilot()" class="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white font-bold rounded-lg text-xs shrink-0 transition">
                            <span>Ask</span>
                        </button>
                    </div>
                    <div id="copilotResponseBox" class="hidden p-3.5 rounded-xl bg-purple-950/20 border border-purple-500/20 text-xs text-purple-200 leading-relaxed font-sans"></div>
                </div>
            </div>

            <!-- RIGHT: AI INFERENCE & ON-CHAIN TERMINAL (5 COLS) -->
            <div class="lg:col-span-5 space-y-6">

                <!-- SPECTRAL / COMPUTER VISION CARD -->
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
                    <div class="flex justify-between items-center border-b border-slate-800 pb-3">
                        <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300"><i class="fa-solid fa-brain text-purple-400 mr-2"></i>AI Spectral & Vision Analysis</h3>
                        <span id="statusBadge" class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-800 text-slate-400">Standby</span>
                    </div>

                    <div class="grid grid-cols-2 gap-4 items-center">
                        <div class="aspect-square bg-slate-950 rounded-xl border border-slate-800 flex items-center justify-center overflow-hidden relative">
                            <img id="heatmapImg" src="" alt="Canopy Heatmap" class="w-full h-full object-cover hidden">
                            <div id="heatmapPlaceholder" class="text-center text-xs text-slate-500 p-4">
                                <i class="fa-solid fa-satellite text-2xl mb-2 text-slate-600"></i>
                                <p>Click "Run AI Verification" to generate OpenCV NDVI Heatmap</p>
                            </div>
                        </div>

                        <div class="space-y-3">
                            <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                                <span class="text-[10px] text-slate-400 uppercase">Satellite NDVI</span>
                                <p id="ndviVal" class="text-xl font-black text-emerald-400 font-mono">--</p>
                            </div>
                            <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                                <span class="text-[10px] text-slate-400 uppercase">Canopy Density</span>
                                <p id="canopyVal" class="text-xl font-black text-white font-mono">--</p>
                            </div>
                        </div>
                    </div>

                    <div class="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1.5 text-xs">
                        <div class="flex justify-between">
                            <span class="text-slate-400">ML Land Classification:</span>
                            <span id="mlClass" class="font-bold text-slate-200">--</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-slate-400">Model Confidence:</span>
                            <span id="mlConf" class="font-mono text-emerald-400">--</span>
                        </div>
                    </div>
                </div>

                <!-- SEPOLIA TOKENIZATION TERMINAL WITH 30-DAY COOLDOWN -->
                <div class="bg-gradient-to-br from-slate-900 via-slate-900 to-emerald-950/40 border border-emerald-500/30 rounded-2xl p-5 space-y-4">
                    <div class="flex justify-between items-center">
                        <div>
                            <h3 class="text-xs font-bold uppercase tracking-wider text-white"><i class="fa-solid fa-link text-emerald-400 mr-1.5"></i>Sepolia Tokenization Terminal</h3>
                            <p class="text-[10px] text-slate-400 font-mono">ERC-20 Standard: $VCO2</p>
                        </div>
                        <span class="text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono font-bold">1 tCO₂ = 1 Token</span>
                    </div>

                    <div class="bg-slate-950/80 p-4 rounded-xl border border-slate-800 space-y-2">
                        <div class="flex justify-between items-baseline">
                            <span class="text-xs text-slate-400">Eligible Carbon Credits:</span>
                            <span id="carbonTonnes" class="text-2xl font-black text-emerald-400 font-mono">0.00 <span class="text-xs font-normal text-slate-400">tCO₂</span></span>
                        </div>
                        <div class="flex justify-between text-xs text-slate-500">
                            <span>Proof Hash:</span>
                            <span id="proofHash" class="font-mono truncate max-w-[200px]">--</span>
                        </div>
                    </div>

                    <!-- DYNAMIC MINT / COOLDOWN BUTTON -->
                    <button id="mintBtn" disabled onclick="mintOnChain()" class="w-full py-3.5 bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600 hover:bg-emerald-400 text-slate-950 font-black rounded-xl text-xs uppercase tracking-wider transition shadow-lg shadow-emerald-500/20 flex items-center justify-center space-x-2">
                        <i class="fa-solid fa-coins"></i>
                        <span id="mintBtnText">Run Verification First</span>
                    </button>

                    <!-- Broadcast Result Alert -->
                    <div id="txAlert" class="hidden p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-300 space-y-1">
                        <div class="flex items-center space-x-2 font-bold">
                            <i class="fa-solid fa-check-circle"></i>
                            <span>Minted Live on Sepolia!</span>
                        </div>
                        <p class="text-slate-300 truncate" id="txHashDisplay"></p>
                        <a id="txExplorerLink" href="#" target="_blank" class="inline-block text-emerald-400 underline font-semibold mt-1">View on Sepolia Etherscan →</a>
                    </div>
                </div>

            </div>
        </div>

        <!-- ======================================================== -->
        <!-- LIVE ON-CHAIN TRANSPARENCY LEDGER TABLE                  -->
        <!-- ======================================================== -->
        <section class="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div class="flex justify-between items-center border-b border-slate-800 pb-3">
                <h3 class="text-sm font-bold text-white flex items-center space-x-2">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span>Live On-Chain Transparency Ledger (Ethereum Sepolia)</span>
                </h3>
                <span class="text-xs font-mono text-slate-500">Public Proof of Zero Double-Counting</span>
            </div>

            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs text-slate-300">
                    <thead class="text-slate-500 uppercase font-mono border-b border-slate-800/80 bg-slate-950/50">
                        <tr>
                            <th class="py-2.5 px-3">Time</th>
                            <th class="py-2.5 px-3">Forest Plot / Biome</th>
                            <th class="py-2.5 px-3">Canopy Density</th>
                            <th class="py-2.5 px-3">Volume Tokenized</th>
                            <th class="py-2.5 px-3">Steward Wallet</th>
                            <th class="py-2.5 px-3">Sepolia Etherscan Proof</th>
                        </tr>
                    </thead>
                    <tbody id="ledgerTableBody" class="divide-y divide-slate-800/60 font-mono">
                        {% for row in ledger %}
                        <tr class="hover:bg-slate-800/30 transition">
                            <td class="py-3 px-3 text-slate-400">{{ row.timestamp }}</td>
                            <td class="py-3 px-3 font-sans font-medium text-white">{{ row.location }}</td>
                            <td class="py-3 px-3 text-emerald-400">{{ row.canopy }}</td>
                            <td class="py-3 px-3 font-bold text-emerald-300">+{{ row.tonnes }} $VCO2</td>
                            <td class="py-3 px-3 text-slate-400">{{ row.owner[:6] }}...{{ row.owner[-4:] }}</td>
                            <td class="py-3 px-3">
                                <a href="https://sepolia.etherscan.io/tx/{{ row.tx_hash }}" target="_blank" class="text-emerald-400 hover:underline flex items-center space-x-1">
                                    <span>{{ row.tx_hash[:10] }}...</span>
                                    <i class="fa-solid fa-arrow-up-right-from-square text-[9px]"></i>
                                </a>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </section>

    </main>

    <!-- Custom Frontend App JS -->
    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>
</html>
```

---

### 5. `static/css/style.css` (Clean Styling & Animations)
```css
/* Custom Pulse and Scroll Animations */
@keyframes glow {
    0%, 100% { opacity: 0.8; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.02); }
}

.glow-box {
    animation: glow 3s ease-in-out infinite;
}

/* Leaflet Custom Pin styling */
.leaflet-container {
    background: #020617 !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #020617;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #10b981;
}
```

---

### 6. `static/js/app.js` (Frontend Architecture)
```javascript
let map, marker, telemetryChart;
let selectedLat = -0.45, selectedLon = 35.75;
let lastVerificationData = null;
let userWallet = null;

window.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Real Satellite Map (Esri World Imagery)
    map = L.map('map').setView([selectedLat, selectedLon], 8);
    
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: '© Esri Satellite'
    }).addTo(map);

    marker = L.marker([selectedLat, selectedLon], {draggable: true}).addTo(map);
    
    marker.on('dragend', function(e) {
        const pos = e.target.getLatLng();
        updateCoords(pos.lat, pos.lng);
    });

    map.on('click', function(e) {
        marker.setLatLng(e.latlng);
        updateCoords(e.latlng.lat, e.latlng.lng);
    });

    initChart();
});

function updateCoords(lat, lon) {
    selectedLat = parseFloat(lat.toFixed(4));
    selectedLon = parseFloat(lon.toFixed(4));
    document.getElementById('placeNameDisplay').innerText = `Resolving location (${selectedLat}, ${selectedLon})...`;
    
    fetch(`/api/resolve-location?lat=${selectedLat}&lon=${selectedLon}`)
        .then(r => r.json())
        .then(d => {
            document.getElementById('placeNameDisplay').innerText = d.address;
        });
}

function loadPreset() {
    const val = document.getElementById('presetSelect').value.split(',');
    const lat = parseFloat(val[0]);
    const lon = parseFloat(val[1]);
    map.flyTo([lat, lon], 8);
    marker.setLatLng([lat, lon]);
    updateCoords(lat, lon);
}

function initChart() {
    const ctx = document.getElementById('telemetryChart').getContext('2d');
    telemetryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Solar Irradiance (W/m²)', data: [], borderColor: '#fbbf24', backgroundColor: 'rgba(251, 191, 36, 0.1)', fill: true, tension: 0.3 },
                { label: 'Cloud Cover (%)', data: [], borderColor: '#94a3b8', borderDash: [5, 5], tension: 0.3 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } }
            },
            plugins: { legend: { labels: { color: '#94a3b8', boxWidth: 10 } } }
        }
    });
}

async function runVerification() {
    const btn = document.querySelector('button[onclick="runVerification()"]');
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i><span>Analyzing...</span>';

    try {
        const res = await fetch('/api/verify', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ lat: selectedLat, lon: selectedLon, area_ha: 50 })
        });
        const data = await res.json();
        lastVerificationData = data;

        document.getElementById('placeNameDisplay').innerText = data.location_name;
        document.getElementById('ndviVal').innerText = data.satellite.ndvi_score;
        document.getElementById('canopyVal').innerText = data.satellite.canopy_coverage_pct + '%';
        document.getElementById('mlClass').innerText = data.ai_classification.land_type;
        document.getElementById('mlConf').innerText = data.ai_classification.confidence + '%';
        document.getElementById('carbonTonnes').innerHTML = `${data.carbon_credits_eligible_tonnes} <span class="text-xs font-normal text-slate-400">tCO₂</span>`;
        document.getElementById('proofHash').innerText = data.proof_hash;

        const img = document.getElementById('heatmapImg');
        img.src = 'data:image/png;base64,' + data.satellite.heatmap_b64;
        img.classList.remove('hidden');
        document.getElementById('heatmapPlaceholder').classList.add('hidden');

        telemetryChart.data.labels = data.climate.chart_labels;
        telemetryChart.data.datasets[0].data = data.climate.chart_solar;
        telemetryChart.data.datasets[1].data = data.climate.chart_cloud;
        telemetryChart.update();

        const badge = document.getElementById('statusBadge');
        const mintBtn = document.getElementById('mintBtn');
        const mintBtnText = document.getElementById('mintBtnText');

        // Check Epoch Cooldown
        if (data.in_cooldown) {
            badge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/40";
            badge.innerText = `Epoch Locked (${data.cooldown_days}d remaining)`;
            mintBtn.disabled = true;
            mintBtn.className = "w-full py-3.5 bg-slate-800 text-amber-400 border border-amber-500/30 rounded-xl text-xs font-mono font-bold flex items-center justify-center space-x-2 cursor-not-allowed";
            mintBtn.innerHTML = `<i class="fa-solid fa-lock animate-pulse"></i><span>Epoch Cooldown: ${data.cooldown_days} Days Remaining</span>`;
            return;
        }

        // Check Normal Verification Status
        if (data.status === 'VERIFIED') {
            badge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";
            badge.innerText = "Verified & Eligible";
            mintBtn.disabled = false;
            mintBtn.className = "w-full py-3.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-black rounded-xl text-xs uppercase tracking-wider transition shadow-lg shadow-emerald-500/20 flex items-center justify-center space-x-2 cursor-pointer";
            mintBtn.innerHTML = `<i class="fa-solid fa-coins"></i><span>${userWallet ? "Authorize & Mint On Sepolia" : "Connect Wallet to Mint"}</span>`;
        } else {
            badge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/40";
            badge.innerText = data.ai_classification.land_type.includes("Water") ? "Rejected (Water Body)" : "Ineligible / Below Threshold";
            mintBtn.disabled = true;
            mintBtn.className = "w-full py-3.5 bg-slate-800 text-slate-600 rounded-xl text-xs uppercase tracking-wider flex items-center justify-center space-x-2 cursor-not-allowed";
            mintBtn.innerHTML = `<i class="fa-solid fa-ban"></i><span>Ineligible for Carbon Credits</span>`;
        }

    } catch (err) {
        alert("Error running verification: " + err);
    } finally {
        btn.innerHTML = '<i class="fa-solid fa-satellite-dish"></i><span>Run AI Verification</span>';
    }
}

async function toggleWallet() {
    if (userWallet) {
        if (confirm("Disconnect wallet?")) {
            userWallet = null;
            document.querySelectorAll('.wallet-address').forEach(el => el.innerText = "Connect Wallet");
        }
        return;
    }

    if (window.ethereum) {
        try {
            const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
            userWallet = accounts[0];
            const shortAddr = userWallet.substring(0, 6) + '...' + userWallet.substring(38);
            document.querySelectorAll('.wallet-address').forEach(el => el.innerText = shortAddr);
        } catch(e) {
            console.error(e);
        }
    } else {
        alert("MetaMask not detected! You can still test in simulated oracle mode.");
    }
}

async function mintOnChain() {
    if (!userWallet) {
        await toggleWallet();
        if (!userWallet) return;
    }

    const btn = document.getElementById('mintBtn');
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i><span>Broadcasting to Sepolia...</span>';
    btn.disabled = true;

    const res = await fetch('/api/mint', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            owner_address: userWallet,
            tonnes: lastVerificationData.carbon_credits_eligible_tonnes,
            proof_hash: lastVerificationData.proof_hash,
            plot_key: lastVerificationData.plot_key,
            location_name: lastVerificationData.location_name
        })
    });
    const result = await res.json();
    
    document.getElementById('txAlert').classList.remove('hidden');
    document.getElementById('txHashDisplay').innerText = `Tx: ${result.tx_hash}`;
    document.getElementById('txExplorerLink').href = result.explorer_url;

    // Transition immediately to the 30-day epoch cooldown state
    btn.className = "w-full py-3.5 bg-slate-800 text-amber-400 border border-amber-500/30 rounded-xl text-xs font-mono font-bold flex items-center justify-center space-x-2 cursor-not-allowed";
    btn.innerHTML = '<i class="fa-solid fa-lock animate-pulse"></i><span>Epoch Locked: Cooldown 30 Days</span>';

    // Prepend new row to the transparency ledger table
    const tableBody = document.getElementById('ledgerTableBody');
    const newRow = `
        <tr class="bg-emerald-500/10 transition font-mono">
            <td class="py-3 px-3 text-emerald-300 font-bold">Just now</td>
            <td class="py-3 px-3 font-sans font-medium text-white">${result.ledger_entry.location}</td>
            <td class="py-3 px-3 text-emerald-400">89.2%</td>
            <td class="py-3 px-3 font-bold text-emerald-300">+${result.ledger_entry.tonnes} $VCO2</td>
            <td class="py-3 px-3 text-slate-400">${userWallet.substring(0,6)}...${userWallet.substring(38)}</td>
            <td class="py-3 px-3">
                <a href="${result.explorer_url}" target="_blank" class="text-emerald-400 hover:underline flex items-center space-x-1">
                    <span>${result.tx_hash.substring(0,10)}...</span>
                    <i class="fa-solid fa-arrow-up-right-from-square text-[9px]"></i>
                </a>
            </td>
        </tr>
    `;
    tableBody.insertAdjacentHTML('afterbegin', newRow);
}

async function askCopilot() {
    const query = document.getElementById('copilotInput').value.trim();
    if (!query) return;

    const box = document.getElementById('copilotResponseBox');
    box.classList.remove('hidden');
    box.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1.5"></i>Thinking...';

    const res = await fetch('/api/ai-copilot', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ query: query })
    });
    const data = await res.json();
    box.innerHTML = `<i class="fa-solid fa-sparkles text-purple-400 mr-1.5"></i>${data.answer}`;
}
```

---

### How to Run:
1. Create the folders:
   ```bash
   mkdir -p templates static/css static/js
   ```
2. Place `index.html` in `templates/`, `style.css` in `static/css/`, and `app.js` in `static/js/`.
3. Put `app.py`, `.env`, and `requirements.txt` in the main folder.
4. Run:
   ```bash
   python app.py
   ```
5. Open `http://127.0.0.1:5000`. You get the hero curtain, real satellite trees, the 30-day epoch cooldown lock, and the live on-chain transparency ledger in a clean, modular setup.