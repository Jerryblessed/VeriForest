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

    marker = L.marker([selectedLat, selectedLon], { draggable: true }).addTo(map);

    marker.on('dragend', function (e) {
        const pos = e.target.getLatLng();
        updateCoords(pos.lat, pos.lng);
    });

    map.on('click', function (e) {
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
            headers: { 'Content-Type': 'application/json' },
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
        } catch (e) {
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

    try {
        const res = await fetch('/api/mint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                owner_address: userWallet,
                tonnes: lastVerificationData.carbon_credits_eligible_tonnes,
                proof_hash: lastVerificationData.proof_hash,
                plot_key: lastVerificationData.plot_key,
                location_name: lastVerificationData.location_name,
                canopy_pct: lastVerificationData.satellite.canopy_coverage_pct + '%'
            })
        });

        const result = await res.json();

        // Check if backend returned an error
        if (!res.ok || result.status === 'ERROR') {
            alert("Minting Failed: " + (result.message || "Unknown error"));
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-coins"></i><span>Authorize & Mint On Sepolia</span>';
            return;
        }

        // 1. Show transaction alert
        document.getElementById('txAlert').classList.remove('hidden');
        document.getElementById('txHashDisplay').innerText = `Tx: ${result.tx_hash}`;
        document.getElementById('txExplorerLink').href = result.explorer_url;

        // 2. Lock button into 30-day epoch cooldown
        btn.className = "w-full py-3.5 bg-slate-800 text-amber-400 border border-amber-500/30 rounded-xl text-xs font-mono font-bold flex items-center justify-center space-x-2 cursor-not-allowed";
        btn.innerHTML = '<i class="fa-solid fa-lock animate-pulse"></i><span>Epoch Locked: Cooldown 30 Days</span>';

        // 3. Update the Live On-Chain Transparency Ledger Table
        const emptyRow = document.getElementById('emptyLedgerRow');
        if (emptyRow) emptyRow.remove();

        const tableBody = document.getElementById('ledgerTableBody');
        const newRow = `
            <tr class="bg-emerald-500/10 transition font-mono">
                <td class="py-3 px-3 text-emerald-300 font-bold">Just now</td>
                <td class="py-3 px-3 font-sans font-medium text-white">${result.ledger_entry.location}</td>
                <td class="py-3 px-3 text-emerald-400">${result.ledger_entry.canopy}</td>
                <td class="py-3 px-3 font-bold text-emerald-300">+${result.ledger_entry.tonnes} $VCO2</td>
                <td class="py-3 px-3 text-slate-400">${userWallet.substring(0, 6)}...${userWallet.substring(38)}</td>
                <td class="py-3 px-3">
                    <a href="${result.explorer_url}" target="_blank" class="text-emerald-400 hover:underline flex items-center space-x-1">
                        <span>${result.tx_hash.substring(0, 10)}...</span>
                        <i class="fa-solid fa-arrow-up-right-from-square text-[9px]"></i>
                    </a>
                </td>
            </tr>
        `;
        tableBody.insertAdjacentHTML('afterbegin', newRow);

    } catch (err) {
        alert("Network Error: " + err);
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-coins"></i><span>Authorize & Mint On Sepolia</span>';
    }
}

async function askCopilot() {
    const query = document.getElementById('copilotInput').value.trim();
    if (!query) return;

    const box = document.getElementById('copilotResponseBox');
    box.classList.remove('hidden');
    box.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1.5"></i>Analyzing query & scanning coordinates...';

    // 1. SMART AGENT NAVIGATION: If user mentions a known biome, fly the map there automatically!
    const qLower = query.toLowerCase();
    let target = null;
    if (qLower.includes("mau") || qLower.includes("kenya")) target = [-0.45, 35.75];
    else if (qLower.includes("amazon") || qLower.includes("brazil")) target = [-3.46, -62.21];
    else if (qLower.includes("borneo") || qLower.includes("malaysia")) target = [1.35, 110.15];
    else if (qLower.includes("black forest") || qLower.includes("germany")) target = [48.20, 8.20];
    else if (qLower.includes("cross river") || qLower.includes("nigeria")) target = [5.80, 8.75];

    if (target) {
        map.flyTo(target, 8, { duration: 1.5 });
        marker.setLatLng(target);
        updateCoords(target[0], target[1]);
        // Trigger live satellite verification automatically
        setTimeout(() => runVerification(), 1600);
    }

    // 2. Fetch the AI response text
    const res = await fetch('/api/ai-copilot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query })
    });
    const data = await res.json();
    box.innerHTML = `<i class="fa-solid fa-sparkles text-purple-400 mr-1.5"></i>${data.answer}`;
}