# 🛰️ VeriForest: Autonomous Satellite AI & Blockchain Carbon dMRV

[![IEEE ClimateChain Hackathon](https://img.shields.io/badge/IEEE-ClimateChain_2026-00629B?style=for-the-badge&logo=ieee&logoColor=white)](https://ieee-climatechain-hack.devpost.com/)
[![Track 1](https://img.shields.io/badge/Track_1-Carbon_Markets_%26_Emissions-10B981?style=for-the-badge)](https://ieee-climatechain-hack.devpost.com/)
[![Network](https://img.shields.io/badge/Network-Ethereum_Sepolia-627EEA?style=for-the-badge&logo=ethereum&logoColor=white)](https://sepolia.etherscan.io/address/0x479CfF5a2d82CFF0C9D255f92842801b41071b65)
[![Smart Contract](https://img.shields.io/badge/ERC--20-$VCO2-purple?style=for-the-badge)](https://sepolia.etherscan.io/address/0x479CfF5a2d82CFF0C9D255f92842801b41071b65)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Autonomous Digital Measurement, Reporting, and Verification (dMRV) protocol eliminating phantom carbon credits through Copernicus Sentinel-2 satellite AI, climate physics models, and immutable Ethereum Sepolia smart contracts.**

---

## 📌 Table of Contents
- [The Problem: The Phantom Credit Crisis](#-the-problem-the-phantom-credit-crisis)
- [The VeriForest Solution](#-the-veriforest-solution)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Smart Contract Specification](#-smart-contract-specification)
- [Technology Stack](#-technology-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started / Installation](#-getting-started--installation)
- [Configuration (.env)](#-configuration-env)
- [API Endpoints](#-api-endpoints)
- [Future Roadmap (Phase 2)](#-future-roadmap-phase-2)
- [License](#-license)

---

## 🚩 The Problem: The Phantom Credit Crisis

In 2023, an extensive investigative report by *The Guardian*, *Die Zeit*, and *SourceMaterial* revealed that **over 90% of certified rainforest carbon offsets approved by the world's leading certifier were "phantom credits"** that did not represent genuine carbon removal.

The traditional carbon certification industry is fundamentally broken:
1. **Exorbitant Costs:** Manual consultant audits cost between **$50,000 and $200,000+**, effectively locking smallholder farmers, indigenous cooperatives, and local conservationists out of global climate finance.
2. **Multi-Year Delays:** Approving a single reforestation plot takes **18 to 36 months**.
3. **Static & Outdated Reports:** Audits occur once every 5 years with static PDF certificates. If a forest is cut down or burns down 6 months after an audit, fake credits continue trading unnoticed for years.
4. **Double-Counting:** Without transparent cryptographic retirement, carbon credits are frequently sold multiple times to different buyers.

---

## 💡 The VeriForest Solution

**VeriForest** replaces slow, expensive, paper-based auditing with an **autonomous satellite AI engine linked to an on-chain smart contract oracle**:

* **Zero Audit Cost:** Leverages open-access Copernicus Sentinel-2 multi-spectral satellite imagery and Open-Meteo real-time climate data.
* **Continuous Sub-Meter Verification:** Scans forest canopies in seconds to quantify living tree biomass, chlorophyll health, and photosynthetic density.
* **Cryptographic Truth:** Mints **`$VCO2`** tokens (1 token = 1 verified tonne of $CO_2$) directly to the land steward's wallet on Ethereum Sepolia.
* **Biological Epoch Cooldown:** Locks verified plots into a **30-day cooldown**, ensuring tokens can only be generated in harmony with biological tree growth cycles.
* **Permanent On-Chain Retirement:** Corporate emitters burn credits on-chain to generate an immutable, tamper-proof Certificate of Net-Zero that eliminates double-counting.

---

## 🏛️ System Architecture

![VeriForest Architecture Diagram](https://github.com/Jerryblessed/VeriForest/blob/main/static/architectural_diagram/veriforest_architecture.png?raw=true)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           1. REMOTE SENSING & CLIMATE LAYER                     │
│  [Copernicus Sentinel-2 Level-2A]               [Open-Meteo Solar Physics API]  │
│  (B02 Blue, B03 Green, B04 Red, B08 NIR, B11)   (Direct Normal Irradiance W/m²) │
└────────────────────────┬────────────────────────────────┬───────────────────────┘
                         │                                │
                         ▼                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             2. AI & COMPUTER VISION ENGINE                      │
│   • Scikit-Learn Random Forest Classifier (Multi-Spectral Land-Cover Model)     │
│   • OpenCV Morphological Canopy Segmentation & NDVI Heatmap Generation         │
│   • Monteith Light-Use Efficiency (LUE) Carbon Sequestration Model             │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼ (Calculates Verified Tonnes of CO2)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            3. WEB3 ORACLE & SMART CONTRACT                      │
│   • Python Web3.py Oracle (Signs & Broadcasts Satellite Proof Hash)             │
│   • VeriForestToken.sol on Ethereum Sepolia (ERC-20, ERC20Burnable)             │
│   • 30-Day Anti-Poaching Biological Epoch Lock                                  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼ (Live Transaction Mining)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         4. USER COCKPIT & TRANSPARENCY LEDGER                   │
│   • Leaflet.js with Sub-Meter Esri World Imagery Satellite Tiles                │
│   • Azure OpenAI (GPT-5.4-nano) Climate Copilot Navigation                     │
│   • Live On-Chain Transparency Ledger with Sepolia Etherscan Proof              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

* **🌍 Real Optical Satellite Imagery:** Integrated with sub-meter Esri World Imagery so users and auditors can visually inspect genuine forest canopies from orbit.
* **🌊 Autonomous Marine & Arid Rejection:** Uses `geopy` reverse-geocoding and spectral indices to detect international waters and deserts. Clicking in the ocean renders a marine colormap and flags the plot as **`Rejected (Water Body)`** with zero tokens eligible.
* **🔬 Multi-Spectral AI Classifier:** Evaluates Near-Infrared (Band 8) and Red (Band 4) wavelengths to accurately compute NDVI:
  $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
* **☀️ Solar Irradiance Integration:** Uses hourly solar radiation ($W/m^2$) to calculate photosynthetic biomass accumulation via the Light-Use Efficiency (LUE) formula.
* **🔒 30-Day Biological Epoch Cooldown:** Plots minted are cryptographically timestamped and locked for 30 days, preventing infinite-token spam.
* **🤖 Azure OpenAI Climate Copilot:** Built-in conversational AI assistant (`gpt-5.4-nano`) that answers carbon methodology questions and automatically navigates the map to requested biomes.
* **📜 Live Transparency Ledger:** Displays on-chain mint events with direct, clickable links to verified transactions on Sepolia Etherscan.

---

## ⛓️ Smart Contract Specification

The **`VeriForestToken ($VCO2)`** contract is deployed live on **Ethereum Sepolia**:

| Parameter | Value |
| :--- | :--- |
| **Contract Name** | `VeriForestToken` |
| **Token Symbol** | `$VCO2` |
| **Standard** | `ERC-20`, `ERC20Burnable`, `Ownable` |
| **Contract Address** | [`0x479CfF5a2d82CFF0C9D255f92842801b41071b65`](https://sepolia.etherscan.io/address/0x479CfF5a2d82CFF0C9D255f92842801b41071b65) |
| **Deployment Tx Hash** | [`0xb5381b783966496b8c611d1ca6918a260797159e299c61d2df088786566d81cf`](https://sepolia.etherscan.io/tx/0xb5381b783966496b8c611d1ca6918a260797159e299c61d2df088786566d81cf) |
| **Token Conversion** | `1 Token = 1 Metric Tonne of verified CO₂ sequestered` |
| **Network** | Ethereum Sepolia Testnet (Chain ID: `11155111`) |

### Core Functions:
* **`mintCredits(address forestOwner, uint256 tonnes, string satelliteProofHash)`**: Only callable by the verified AI Oracle backend. Mints tokens to the verified land steward and logs the cryptographic proof hash.
* **`retireCredits(uint256 tonnes, string companyName, string reason)`**: Permanently burns tokens out of circulation and emits an on-chain `CarbonRetired` event, preventing double-spending.

---

## 🛠️ Technology Stack

* **Backend & Oracles:** Python 3.10+, Flask, `web3.py`, `python-dotenv`, `requests`
* **AI & Computer Vision:** `opencv-python-headless`, `scikit-learn`, `numpy`
* **Smart Contracts & Blockchain:** Solidity 0.8.20, OpenZeppelin Contracts, Remix IDE, MetaMask
* **Geospatial & Climate APIs:** `geopy` (Nominatim OpenStreetMap), Esri World Imagery, Open-Meteo Solar Forecast API
* **Large Language Model:** Azure OpenAI SDK (`gpt-5.4-nano`)
* **Frontend UI:** HTML5, CSS3, Tailwind CSS, JavaScript (ES6+), Leaflet.js, Chart.js, FontAwesome

---

## 📁 Project Directory Structure

```text
veriforest/
├── .env                              # Environment secrets (RPC, Contract, Keys)
├── requirements.txt                  # Python dependency specifications
├── app.py                            # Flask application controller & AI engine
├── contracts/
│   └── VeriForestToken.sol           # Solidity ERC-20 smart contract
├── templates/
│   └── index.html                    # Single-page cockpit & hero template
└── static/
    ├── css/
    │   └── style.css                 # Custom animations and scrollbars
    └── js/
        └── app.js                    # Leaflet map, Chart.js, & Web3 frontend logic
```

---

## 🚀 Getting Started / Installation

### Prerequisites
* Python 3.10 or higher
* Git
* A Web3 wallet (e.g., [MetaMask](https://metamask.io/)) configured for Ethereum Sepolia

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/veriforest.git
cd veriforest
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory (see [Configuration](#-configuration-env) below).

### 5. Launch the Application
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## ⚙️ Configuration (`.env`)

Create a `.env` file in the root folder with the following parameters:

```env
# Ethereum Sepolia RPC Endpoint
RPC_URL=https://rpc.sepolia.org

# Deployed Smart Contract Address
CONTRACT_ADDRESS=0x479CfF5a2d82CFF0C9D255f92842801b41071b65

# 64-character Oracle Backend Private Key (used to sign mint transactions)
ORACLE_PRIVATE_KEY=your_64_character_hex_private_key_without_0x

# Azure OpenAI Configuration (Optional - built-in fallback active if blank)
AZURE_OPENAI_ENDPOINT=https://opejeremiah-2939-resource.services.ai.azure.com/openai/v1
AZURE_OPENAI_DEPLOYMENT=gpt-5.4-nano
AZURE_OPENAI_KEY=your_azure_openai_key
```

*(⚠️ **Security Warning:** Never commit your `.env` file or private keys to a public GitHub repository. The `.env` file is excluded via `.gitignore`.)*

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the main landing hero curtain and interactive dMRV terminal. |
| `GET` | `/api/resolve-location` | Reverse-geocodes `lat` and `lon` to extract human-readable place names and water flags. |
| `POST` | `/api/verify` | Runs the AI multi-spectral classifier, OpenCV heatmap segmentation, and solar LUE carbon calculation. |
| `POST` | `/api/mint` | Triggers the Web3 Python Oracle to sign and broadcast the mint transaction on Sepolia. |
| `POST` | `/api/ai-copilot` | Natural language agent endpoint answering carbon market and MRV methodology queries. |

---

## 🗺️ Future Roadmap (Phase 2)

While our hackathon MVP demonstrates autonomous satellite dMRV and on-chain tokenization, our production roadmap includes:

1. **Decentralized Land Title Verification (Soulbound Tokens / DIDs):**
   * Integrate Soulbound Tokens (SBTs) to bind official government land titles (e.g., Certificate of Occupancy, Title Deeds, Concession Agreements) to landholder wallets prior to plot activation.
2. **Multi-Point GeoJSON Cadastral Shapefiles:**
   * Transition from coordinate radius boxes to sub-meter precision cadastral polygons directly integrated with national spatial databases (e.g., Brazil’s CAR, Kenya’s Land Commission).
3. **Automated Deforestation Slashing:**
   * If periodic satellite passes detect unauthorized clear-cutting or wildfire destruction mid-epoch, an automated oracle slashes the plot's staking deposit and halts future token issuance.
4. **P2P Decentralized Carbon Marketplace:**
   * Launch atomic swap liquidity pools allowing heavy industrial emitters to purchase and burn `$VCO2` credits directly using stablecoins (e.g., USDC), completely eliminating broker commission fees.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🏆 Hackathon Attribution
Built with pride for the **IEEE ClimateChain Global Hackathon 2026** in support of UN COP31 climate initiatives.  
*Track 1: Carbon Markets & Emissions Transparency.*
