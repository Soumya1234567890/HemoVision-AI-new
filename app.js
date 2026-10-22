/**
 * HemoVision AI — Clinical Hemoglobinopathy Diagnostic & Multi-Perspective XAI
 * Pure Vanilla JavaScript Client Logic (Version 2.0.0)
 */

// Application State
const state = {
  modelInfo: null,
  patientData: {},
  latestPrediction: null,
  cohortData: [],
  boundaryData: null,
  rbcChart: null,
  rdwChart: null
};

// DOM Elements
const patientForm = document.getElementById('patientForm');
const mentzerDisplay = document.getElementById('mentzerDisplay');
const mentzerPill = document.getElementById('mentzerPill');
const btnRunInference = document.getElementById('btnRunInference');
const btnResetInputs = document.getElementById('btnResetInputs');
const presetSelect = document.getElementById('presetSelect');
const btnRandomCohort = document.getElementById('btnRandomCohort');
const btnExportReport = document.getElementById('btnExportReport');
const reportModal = document.getElementById('reportModal');
const btnCloseModal = document.getElementById('btnCloseModal');

// Canvas Elements
const canvas = document.getElementById('decisionCanvas');
const canvasWrapper = document.getElementById('canvasWrapper');
const canvasTooltip = document.getElementById('canvasTooltip');
const axisXSelect = document.getElementById('axisX');
const axisYSelect = document.getElementById('axisY');
const btnRecalculateBoundary = document.getElementById('btnRecalculateBoundary');

// Diagnosis helper matching
function isBttDiagnosis(diag) {
  if (!diag) return false;
  const s = String(diag).toLowerCase();
  return s.includes('beta') || s.includes('btt');
}

function isIdaDiagnosis(diag) {
  if (!diag) return false;
  const s = String(diag).toLowerCase();
  return (s.includes('iron') && s.includes('anemia')) || s.includes('ida');
}

// ---------------------------------------------------------
// 1. INITIALIZE APPLICATION
// ---------------------------------------------------------
async function initApp() {
  try {
    // 1. Fetch Model Metadata
    const infoRes = await fetch('/api/info');
    state.modelInfo = await infoRes.json();

    // 2. Build Patient Input Form
    buildInputControls(state.modelInfo.features);

    // 3. Render Static Model Validation Info (Tab 7)
    renderValidationAudit(state.modelInfo);

    // 4. Load Clinical Presets
    loadPresets();

    // 5. Fetch Cohort Data
    loadCohort('ida_btt');

    // 6. Setup Tab Navigation
    setupTabs();

    // 7. Setup Canvas Event Listeners
    setupCanvas();

    // 8. Initial Inference Run
    await runInference();

  } catch (err) {
    console.error('Initialization error:', err);
  }
}

// Dynamically construct input sliders & numeric inputs
function buildInputControls(features) {
  patientForm.innerHTML = '';
  
  for (const [key, meta] of Object.entries(features)) {
    // Mentzer Index is auto-computed from MCV / RBC
    if (key === 'Mentzer Index') continue;

    state.patientData[key] = meta.default;

    const cleanId = key.replace(/[^a-zA-Z0-9]/g, '_');
    const item = document.createElement('div');
    item.className = 'feature-item';
    item.id = `item_${cleanId}`;

    item.innerHTML = `
      <div class="feature-header">
        <div>
          <span class="feature-label">${meta.label}</span>
          <span class="feature-unit">(${meta.unit})</span>
        </div>
        ${key !== 'Age_years' && meta.normal_range ? `
        <span id="range_badge_${cleanId}" class="ref-badge normal">
          Ref: ${meta.normal_range[0]} - ${meta.normal_range[1]}
        </span>` : ''}
      </div>
      <div class="feature-input-group">
        <input type="range" class="feature-slider" 
               id="slider_${cleanId}"
               min="${meta.min}" max="${meta.max}" step="${meta.step}" value="${meta.default}">
        <input type="number" class="feature-num-input" 
               id="input_${cleanId}"
               min="${meta.min}" max="${meta.max}" step="${meta.step}" value="${meta.default}">
      </div>
      <div class="ref-range-info">
        <span>Min: ${meta.min}</span>
        <span>Max: ${meta.max}</span>
      </div>
    `;

    patientForm.appendChild(item);

    const slider = item.querySelector('.feature-slider');
    const numInput = item.querySelector('.feature-num-input');

    // Sync Slider and Number Input
    slider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      numInput.value = val;
      state.patientData[key] = val;
      updateInputRangeBadge(key, val, meta.normal_range);
      updateCalculatedMentzer();
    });

    numInput.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (!isNaN(val)) {
        slider.value = val;
        state.patientData[key] = val;
        updateInputRangeBadge(key, val, meta.normal_range);
        updateCalculatedMentzer();
      }
    });

    // Auto-inference on slider release for smooth interactive experience
    slider.addEventListener('change', () => {
      runInference();
    });

    updateInputRangeBadge(key, meta.default, meta.normal_range);
  }

  updateCalculatedMentzer();
}

function updateInputRangeBadge(key, val, range) {
  if (key === 'Age_years' || !range || !Array.isArray(range)) return;
  const cleanId = key.replace(/[^a-zA-Z0-9]/g, '_');
  const badge = document.getElementById(`range_badge_${cleanId}`);
  if (!badge) return;

  if (val < range[0]) {
    badge.className = 'ref-badge low';
    badge.textContent = `LOW (${range[0]} - ${range[1]})`;
  } else if (val > range[1]) {
    badge.className = 'ref-badge high';
    badge.textContent = `HIGH (${range[0]} - ${range[1]})`;
  } else {
    badge.className = 'ref-badge normal';
    badge.textContent = `Normal (${range[0]} - ${range[1]})`;
  }
}

// Compute Mentzer Index (MCV / RBC) in real time
function updateCalculatedMentzer() {
  const mcv = state.patientData['MCV'] || 64.0;
  const rbc = state.patientData['RBC Count'] || 5.5;
  const mentzer = rbc > 0 ? (mcv / rbc).toFixed(1) : '13.0';

  state.patientData['Mentzer Index'] = parseFloat(mentzer);
  mentzerDisplay.textContent = mentzer;

  if (parseFloat(mentzer) < 13.0) {
    mentzerPill.className = 'mentzer-status-pill btt';
    mentzerPill.textContent = 'Rule: <13 (Suggests BTT)';
  } else {
    mentzerPill.className = 'mentzer-status-pill ida';
    mentzerPill.textContent = 'Rule: ≥13 (Suggests IDA)';
  }
}

// ---------------------------------------------------------
// 2. DIAGNOSTIC INFERENCE RUNNER
// ---------------------------------------------------------
async function runInference() {
  try {
    updateCalculatedMentzer();

    const payload = {
      Age_years: state.patientData['Age_years'],
      "RBC Count": state.patientData['RBC Count'],
      HCT: state.patientData['HCT'],
      Hemoglobin: state.patientData['Hemoglobin'],
      MCV: state.patientData['MCV'],
      MCH: state.patientData['MCH'],
      MCHC: state.patientData['MCHC'],
      "RDW-CV": state.patientData['RDW-CV'],
      "Mentzer Index": state.patientData['Mentzer Index']
    };

    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error('Inference API call failed');
    const result = await res.json();
    state.latestPrediction = result;

    // Update Verdict Hero Card
    updateVerdictUI(result);

    // Update Tab 1: SHAP Waterfall & Odds Ratios
    renderShapWaterfall(result.shap_analysis, result.clinical_narratives);
    renderOddsRatios(state.modelInfo?.global_importance || []);

    // Update Tab 2: 2D Decision Space Surface
    renderDecisionBoundary();

    // Update Tab 3: 5-Rule Concordance Matrix
    renderConcordanceMatrix(result.discriminant_matrix, result.discriminant_agreement_pct);

    // Update Tab 4: Nearest Boundary Exemplars
    renderExemplars(result.nearest_support_vectors);

    // Update Tab 5: What-If Sensitivity Charts & Counterfactuals
    renderSensitivity(result.sensitivity, result.diagnosis);
    renderInversions(result.counterfactual_inversions);

  } catch (err) {
    console.error('Inference error:', err);
  }
}

// ---------------------------------------------------------
// 3. VERDICT HERO CARD UI
// ---------------------------------------------------------
function updateVerdictUI(pred) {
  const hero = document.getElementById('verdictHero');
  const title = document.getElementById('verdictTitle');
  const conf = document.getElementById('verdictConfidence');
  const margin = document.getElementById('verdictMargin');
  const zoneBadge = document.getElementById('marginZoneBadge');
  const probBTT = document.getElementById('probValBTT');
  const probIDA = document.getElementById('probValIDA');
  const barBTT = document.getElementById('probBarBTT');
  const barIDA = document.getElementById('probBarIDA');
  const discordanceBanner = document.getElementById('discordanceBanner');

  const isBTT = pred.predicted_class === 1;

  hero.className = `verdict-hero ${isBTT ? 'btt-verdict' : 'ida-verdict'}`;
  title.textContent = pred.diagnosis;
  conf.textContent = `${pred.confidence}%`;
  conf.style.color = isBTT ? '#c4b5fd' : '#67e8f9';

  const sign = pred.decision_margin >= 0 ? '+' : '';
  margin.textContent = `${sign}${pred.decision_margin}`;

  zoneBadge.textContent = pred.margin_zone;
  zoneBadge.style.background = isBTT ? 'rgba(139, 92, 246, 0.2)' : 'rgba(6, 182, 212, 0.2)';
  zoneBadge.style.color = isBTT ? '#c4b5fd' : '#67e8f9';

  probBTT.textContent = `${pred.probabilities.btt}%`;
  probIDA.textContent = `${pred.probabilities.ida}%`;
  barBTT.style.width = `${pred.probabilities.btt}%`;
  barIDA.style.width = `${pred.probabilities.ida}%`;

  // Mentzer Concordance Banner
  const m = pred.mentzer_comparison;
  if (m.status === 'CONCORDANT') {
    discordanceBanner.className = 'discordance-banner concordant';
    discordanceBanner.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
      <div><strong>Clinical Concordance:</strong> ${m.explanation}</div>
    `;
  } else {
    discordanceBanner.className = 'discordance-banner discordant';
    discordanceBanner.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <div><strong>Clinical Discordance Warning:</strong> ${m.explanation}</div>
    `;
  }
}

// ---------------------------------------------------------
// 4. TAB 1: SHAP WATERFALL & ODDS RATIOS
// ---------------------------------------------------------
function renderShapWaterfall(shapData, narratives) {
  const container = document.getElementById('waterfallContainer');
  const baselineElem = document.getElementById('shapBaseline');
  const narrativeList = document.getElementById('clinicalNarrativeList');

  baselineElem.textContent = `${shapData.baseline_prob}% (Intercept β₀: ${shapData.intercept})`;
  container.innerHTML = '';

  shapData.waterfall.forEach(item => {
    const isBtt = item.direction === 'btt';
    const absScore = Math.abs(item.log_odds_impact);
    const barWidth = Math.min(absScore * 28.0, 48); // scale log-odds for visual bar clarity

    const row = document.createElement('div');
    row.className = 'waterfall-bar-item';
    row.innerHTML = `
      <div class="waterfall-feat-name">${item.label}</div>
      <div class="waterfall-feat-val">${item.patient_value} ${item.unit}</div>
      <div class="waterfall-visual-track">
        <div class="waterfall-center-line"></div>
        <div class="waterfall-bar ${isBtt ? 'push-btt' : 'push-ida'}" 
             style="width: ${Math.max(barWidth, 2)}%;"></div>
      </div>
      <div class="waterfall-score ${isBtt ? 'push-btt' : 'push-ida'}">
        ${isBtt ? '+' : ''}${item.log_odds_impact} <span style="font-size: 0.65rem; opacity: 0.75;">log-odds</span>
      </div>
    `;
    container.appendChild(row);
  });

  narrativeList.innerHTML = '';
  narratives.forEach(text => {
    const li = document.createElement('li');
    li.textContent = text;
    narrativeList.appendChild(li);
  });
}

function renderOddsRatios(importanceList) {
  const tbody = document.getElementById('oddsRatioTableBody');
  if (!tbody || !importanceList.length) return;
  tbody.innerHTML = '';

  importanceList.forEach(item => {
    const tr = document.createElement('tr');
    const isBtt = item.favors === 'Beta Thalassemia Trait';
    const directionColor = isBtt ? '#c084fc' : '#38bdf8';

    tr.innerHTML = `
      <td><strong>${item.label}</strong> <span style="color: var(--text-muted); font-size: 0.7rem;">(${item.unit})</span></td>
      <td><code style="color: ${item.weight >= 0 ? '#c084fc' : '#38bdf8'};">${item.weight >= 0 ? '+' : ''}${item.weight}</code></td>
      <td style="font-weight: 700; color: #f8fafc;">${item.odds_ratio_std}x</td>
      <td style="color: var(--text-secondary);">${item.odds_ratio_unit}x</td>
      <td style="color: ${directionColor}; font-weight: 600;">${item.favors}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------
// 5. TAB 2: 2D DECISION SPACE CANVAS
// ---------------------------------------------------------
function setupCanvas() {
  function resizeCanvas() {
    canvas.width = canvasWrapper.clientWidth;
    canvas.height = canvasWrapper.clientHeight;
    if (state.boundaryData) drawBoundaryCanvas(state.boundaryData);
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  axisXSelect.addEventListener('change', () => renderDecisionBoundary());
  axisYSelect.addEventListener('change', () => renderDecisionBoundary());
  btnRecalculateBoundary.addEventListener('click', () => renderDecisionBoundary());

  // Interactive Hover Inspection on Canvas
  canvas.addEventListener('mousemove', (e) => {
    if (!state.boundaryData) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const b = state.boundaryData;
    const padding = 50;
    const plotW = canvas.width - padding * 2;
    const plotH = canvas.height - padding * 2;

    if (mx < padding || mx > canvas.width - padding || my < padding || my > canvas.height - padding) {
      canvasTooltip.style.display = 'none';
      return;
    }

    const normX = (mx - padding) / plotW;
    const normY = 1.0 - ((my - padding) / plotH);

    const valX = (b.x_range[0] + normX * (b.x_range[1] - b.x_range[0])).toFixed(2);
    const valY = (b.y_range[0] + normY * (b.y_range[1] - b.y_range[0])).toFixed(2);

    let nearbyText = '';
    b.support_vectors.forEach(sv => {
      const svPx = padding + ((sv.x - b.x_range[0]) / (b.x_range[1] - b.x_range[0])) * plotW;
      const svPy = padding + (1.0 - (sv.y - b.y_range[0]) / (b.y_range[1] - b.y_range[0])) * plotH;
      const dist = Math.hypot(mx - svPx, my - svPy);
      if (dist < 8) {
        nearbyText = `<br><span style="color: #c4b5fd;">★ Boundary Exemplar: ${sv.id} (${sv.class_label})</span>`;
      }
    });

    canvasTooltip.style.display = 'block';
    canvasTooltip.style.left = `${mx + 15}px`;
    canvasTooltip.style.top = `${my - 15}px`;
    canvasTooltip.innerHTML = `
      <strong>${b.x_feature}:</strong> ${valX}<br>
      <strong>${b.y_feature}:</strong> ${valY}
      ${nearbyText}
    `;
  });

  canvas.addEventListener('mouseleave', () => {
    canvasTooltip.style.display = 'none';
  });
}

async function renderDecisionBoundary() {
  try {
    const xFeat = axisXSelect.value;
    const yFeat = axisYSelect.value;
    const p = state.patientData;

    const url = `/api/decision-boundary?x_feature=${encodeURIComponent(xFeat)}&y_feature=${encodeURIComponent(yFeat)}&age=${p['Age_years']}&rbc=${p['RBC Count']}&hct=${p['HCT']}&hb=${p['Hemoglobin']}&mcv=${p['MCV']}&mch=${p['MCH']}&mchc=${p['MCHC']}&rdw=${p['RDW-CV']}&grid_size=28`;

    const res = await fetch(url);
    if (!res.ok) throw new Error('Boundary grid API error');
    const bData = await res.json();
    state.boundaryData = bData;

    drawBoundaryCanvas(bData);
  } catch (err) {
    console.error('Error rendering boundary:', err);
  }
}

function drawBoundaryCanvas(data) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const padding = 50;

  ctx.clearRect(0, 0, w, h);

  const plotW = w - padding * 2;
  const plotH = h - padding * 2;

  const xMin = data.x_range[0];
  const xMax = data.x_range[1];
  const yMin = data.y_range[0];
  const yMax = data.y_range[1];

  const toPxX = (val) => padding + ((val - xMin) / (xMax - xMin)) * plotW;
  const toPxY = (val) => padding + (1.0 - (val - yMin) / (yMax - yMin)) * plotH;

  // 1. Render Contour Grid Cells
  const gridN = data.grid_size;
  const cellW = plotW / (gridN - 1);
  const cellH = plotH / (gridN - 1);

  for (let i = 0; i < gridN - 1; i++) {
    for (let j = 0; j < gridN - 1; j++) {
      const prob = data.probs_btt[j][i];

      // Interpolate colors: BTT purple (#8b5cf6) vs IDA cyan (#06b6d4)
      const r = Math.round(6 + prob * (139 - 6));
      const g = Math.round(182 + prob * (92 - 182));
      const bColor = Math.round(212 + prob * (246 - 212));
      const alpha = 0.35 + Math.abs(prob - 0.5) * 0.45;

      ctx.fillStyle = `rgba(${r}, ${g}, ${bColor}, ${alpha})`;

      const px = toPxX(data.x_vals[i]);
      const py = toPxY(data.y_vals[j + 1]);

      ctx.fillRect(px, py, cellW + 1, cellH + 1);
    }
  }

  // 2. Draw Decision Boundary Line (P ~ 0.50)
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);

  for (let i = 0; i < gridN - 1; i++) {
    for (let j = 0; j < gridN - 1; j++) {
      const p1 = data.probs_btt[j][i];
      const p2 = data.probs_btt[j][i + 1];

      if ((p1 - 0.5) * (p2 - 0.5) < 0) {
        const xMid = (data.x_vals[i] + data.x_vals[i + 1]) / 2;
        const yPos = data.y_vals[j];
        ctx.beginPath();
        ctx.arc(toPxX(xMid), toPxY(yPos), 1.5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }
    }
  }
  ctx.setLineDash([]);

  // 3. Draw Grid Lines and Axes
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.lineWidth = 1;
  ctx.strokeRect(padding, padding, plotW, plotH);

  // Axis Labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`${data.x_feature}`, padding + plotW / 2, h - 15);

  ctx.save();
  ctx.translate(15, padding + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(`${data.y_feature}`, 0, 0);
  ctx.restore();

  // Range text
  ctx.textAlign = 'left';
  ctx.fillText(xMin.toFixed(1), padding, h - 35);
  ctx.textAlign = 'right';
  ctx.fillText(xMax.toFixed(1), padding + plotW, h - 35);

  ctx.textAlign = 'left';
  ctx.fillText(yMax.toFixed(1), padding + 5, padding + 15);
  ctx.fillText(yMin.toFixed(1), padding + 5, padding + plotH - 5);

  // 4. Render Historical Cohort Points (Correctly check BTT vs IDA)
  data.cohort_points.forEach(pt => {
    const px = toPxX(pt.x);
    const py = toPxY(pt.y);
    const isBtt = isBttDiagnosis(pt.true_diagnosis);

    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = isBtt ? 'rgba(196, 181, 253, 0.85)' : 'rgba(103, 232, 249, 0.85)';
    ctx.fill();
  });

  // 5. Render Boundary Anchors / Exemplars
  data.support_vectors.forEach(sv => {
    const px = toPxX(sv.x);
    const py = toPxY(sv.y);
    const isBtt = sv.class_code === 1;

    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = isBtt ? '#8b5cf6' : '#06b6d4';
    ctx.fill();
  });

  // 6. Render Active Patient Pin (Red Pulsing Pin)
  const patientX = state.patientData[data.x_feature];
  const patientY = state.patientData[data.y_feature];

  if (patientX !== undefined && patientY !== undefined) {
    const px = toPxX(patientX);
    const py = toPxY(patientY);

    // Glowing Outer Halo
    ctx.beginPath();
    ctx.arc(px, py, 14, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(244, 63, 94, 0.25)';
    ctx.fill();

    // Intermediate Ring
    ctx.beginPath();
    ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Solid Center
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();

    // Label
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Patient Target', px + 12, py + 4);
  }
}

// ---------------------------------------------------------
// 6. TAB 3: 5-RULE CONCORDANCE MATRIX
// ---------------------------------------------------------
function renderConcordanceMatrix(matrix, agreementPct) {
  const tbody = document.getElementById('concordanceTableBody');
  const badge = document.getElementById('concordanceRateBadge');
  if (!tbody || !matrix) return;

  tbody.innerHTML = '';
  badge.textContent = `Rule Agreement: ${agreementPct}%`;
  badge.className = agreementPct >= 80 ? 'badge badge-success' : 'badge badge-warning';

  const aiPredBtt = state.latestPrediction?.predicted_class === 1;

  matrix.forEach(row => {
    const agrees = (row.btt_favored && aiPredBtt) || (!row.btt_favored && !aiPredBtt);
    const tr = document.createElement('tr');

    tr.innerHTML = `
      <td><strong>${row.name}</strong></td>
      <td><code style="color: #94a3b8; font-size: 0.75rem;">${row.formula}</code></td>
      <td style="font-weight: 700; color: #f8fafc;">${row.value}</td>
      <td style="color: var(--text-secondary); font-size: 0.75rem;">${row.cutoff}</td>
      <td style="font-weight: 600; color: ${row.btt_favored ? '#c084fc' : '#38bdf8'};">${row.suggested}</td>
      <td>
        <span class="badge ${agrees ? 'badge-success' : 'badge-danger'}">
          ${agrees ? 'CONCORDANT' : 'DISCORDANT'}
        </span>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------
// 7. TAB 4: BOUNDARY EXEMPLARS (CASE-BASED REASONING)
// ---------------------------------------------------------
function renderExemplars(exemplars) {
  const grid = document.getElementById('exemplarGrid');
  grid.innerHTML = '';

  exemplars.forEach((sv, idx) => {
    const isBtt = sv.class_code === 1;
    const card = document.createElement('div');
    card.className = `exemplar-card ${isBtt ? 'btt' : 'ida'}`;

    card.innerHTML = `
      <div class="exemplar-header">
        <div>
          <span class="exemplar-tag">${sv.diagnosis}</span>
          <div style="font-size: 0.7rem; color: var(--text-muted);">${sv.sv_id} (Rank #${idx + 1} Neighbor)</div>
        </div>
        <div class="exemplar-sim">
          ${sv.similarity_pct}% Similarity
        </div>
      </div>
      <div class="exemplar-metrics">
        <span>RBC: <strong>${sv.features['RBC Count']}</strong> 10¹²/L</span>
        <span>Hb: <strong>${sv.features['Hemoglobin']}</strong> g/dL</span>
        <span>MCV: <strong>${sv.features['MCV']}</strong> fL</span>
        <span>RDW-CV: <strong>${sv.features['RDW-CV']}</strong>%</span>
        <span>Mentzer: <strong>${sv.features['Mentzer Index']}</strong></span>
        <span>Distance: <strong>${sv.distance}</strong></span>
      </div>
    `;
    grid.appendChild(card);
  });
}

// ---------------------------------------------------------
// 8. TAB 5: WHAT-IF SENSITIVITY & COUNTERFACTUALS
// ---------------------------------------------------------
function renderSensitivity(sensitivity, currentDiagnosis) {
  const rbcCtx = document.getElementById('chartRbcSensitivity').getContext('2d');
  const rdwCtx = document.getElementById('chartRdwSensitivity').getContext('2d');
  const cfText = document.getElementById('counterfactualText');

  // RBC Chart
  const rbcLabels = sensitivity.rbc_sweep.map(d => `${d.rbc} (${d.delta > 0 ? '+' : ''}${d.delta})`);
  const rbcProbs = sensitivity.rbc_sweep.map(d => d.new_btt_prob);

  if (state.rbcChart) state.rbcChart.destroy();
  state.rbcChart = new Chart(rbcCtx, {
    type: 'line',
    data: {
      labels: rbcLabels,
      datasets: [{
        label: 'BTT Probability (%)',
        data: rbcProbs,
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139, 92, 246, 0.15)',
        tension: 0.3,
        fill: true,
        pointBackgroundColor: '#8b5cf6',
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        x: {
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });

  // RDW Chart
  const rdwLabels = sensitivity.rdw_sweep.map(d => `${d.rdw} (${d.delta > 0 ? '+' : ''}${d.delta}%)`);
  const rdwProbs = sensitivity.rdw_sweep.map(d => d.new_btt_prob);

  if (state.rdwChart) state.rdwChart.destroy();
  state.rdwChart = new Chart(rdwCtx, {
    type: 'line',
    data: {
      labels: rdwLabels,
      datasets: [{
        label: 'BTT Probability (%)',
        data: rdwProbs,
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.15)',
        tension: 0.3,
        fill: true,
        pointBackgroundColor: '#06b6d4',
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        x: {
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });

  // Dynamic Clinical Counterfactual Inversion narrative
  if (currentDiagnosis.includes('Beta Thalassemia')) {
    cfText.innerHTML = `
      Active determination is <strong>Beta Thalassemia Trait</strong>. Based on multivariate gradient sensitivity, to invert the classification toward <strong>Iron Deficiency Anemia</strong>, the patient's RBC Count would need to drop below the threshold or RDW-CV would need to increase above physiological anisocytosis limits.
    `;
  } else {
    cfText.innerHTML = `
      Active determination is <strong>Iron Deficiency Anemia</strong>. To invert the classification toward <strong>Beta Thalassemia Trait</strong>, the patient's RBC Count would need to rise into the erythrocytosis range (>5.4 10¹²/L) or RDW-CV would need to decline below 16.0% (homogeneous red cell distribution).
    `;
  }
}

function renderInversions(inversions) {
  const grid = document.getElementById('inversionGrid');
  if (!grid || !inversions) return;
  grid.innerHTML = '';

  inversions.forEach(item => {
    const card = document.createElement('div');
    card.className = 'inversion-card';
    const isPlausible = item.plausible;

    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
        <span style="font-weight: 700; color: #f8fafc; font-size: 0.82rem;">${item.label}</span>
        <span class="badge ${isPlausible ? 'badge-primary' : 'badge-outline'}" style="font-size: 0.65rem;">
          ${isPlausible ? 'Physiologically Attainable' : 'Extreme Outlier'}
        </span>
      </div>
      <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.4rem;">
        Current: <strong style="color: #cbd5e1;">${item.current_val} ${item.unit}</strong> → Boundary: <strong style="color: #a5b4fc;">${item.threshold_val} ${item.unit}</strong>
      </div>
      <div style="font-size: 0.72rem; color: ${item.delta_needed > 0 ? '#c084fc' : '#38bdf8'}; font-weight: 600;">
        Requires ${item.direction} of ${Math.abs(item.delta_needed)} ${item.unit}
      </div>
    `;
    grid.appendChild(card);
  });
}

// ---------------------------------------------------------
// 9. TAB 6: HPLC COHORT EXPLORER (1,190 PATIENTS)
// ---------------------------------------------------------
async function loadCohort(filter) {
  try {
    const res = await fetch(`/api/cohort?filter_type=${filter}`);
    const data = await res.json();
    state.cohortData = data.records;

    const tbody = document.getElementById('cohortTableBody');
    tbody.innerHTML = '';

    data.records.forEach(row => {
      const tr = document.createElement('tr');
      const isBtt = isBttDiagnosis(row.hplc_diagnosis);
      const isIda = isIdaDiagnosis(row.hplc_diagnosis);
      const isComb = row.hplc_diagnosis.toLowerCase().includes('combined');
      const isNorm = row.hplc_diagnosis.toLowerCase().includes('normal');

      let badgeColor = '#94a3b8';
      if (isBtt) badgeColor = '#c4b5fd';
      else if (isIda) badgeColor = '#67e8f9';
      else if (isComb) badgeColor = '#fcd34d';
      else if (isNorm) badgeColor = '#6ee7b7';

      tr.innerHTML = `
        <td><strong>${row.id}</strong></td>
        <td>${row.age}</td>
        <td>${row.rbc}</td>
        <td>${row.hb}</td>
        <td>${row.mcv}</td>
        <td>${row.mch}</td>
        <td>${row.rdw}</td>
        <td>${row.mentzer}</td>
        <td style="color: ${badgeColor}; font-weight: 600;">${row.hplc_diagnosis}</td>
        <td>${row.svm_prob_btt !== null ? `${row.svm_prob_btt}% BTT` : '-'}</td>
        <td>
          <button class="cohort-btn-load" data-id="${row.id}">Load into Analyzer</button>
        </td>
      `;

      tr.querySelector('.cohort-btn-load').addEventListener('click', () => {
        loadPatientFromRow(row);
      });

      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Error loading cohort:', err);
  }
}

function loadPatientFromRow(row) {
  state.patientData['Age_years'] = row.age_years;
  state.patientData['RBC Count'] = row.rbc;
  state.patientData['HCT'] = row.hct;
  state.patientData['Hemoglobin'] = row.hb;
  state.patientData['MCV'] = row.mcv;
  state.patientData['MCH'] = row.mch;
  state.patientData['MCHC'] = row.mchc;
  state.patientData['RDW-CV'] = row.rdw;

  // Update inputs
  for (const [key, val] of Object.entries(state.patientData)) {
    if (key === 'Mentzer Index') continue;
    const cleanId = key.replace(/[^a-zA-Z0-9]/g, '_');
    const slider = document.getElementById(`slider_${cleanId}`);
    const input = document.getElementById(`input_${cleanId}`);
    if (slider) slider.value = val;
    if (input) input.value = val;
    const meta = state.modelInfo?.features[key];
    if (meta) updateInputRangeBadge(key, val, meta.normal_range);
  }

  updateCalculatedMentzer();
  runInference();

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ---------------------------------------------------------
// 10. TAB 7: MODEL VALIDATION AUDIT
// ---------------------------------------------------------
function renderValidationAudit(info) {
  const breakdownList = document.getElementById('diagnosisBreakdownList');
  if (!breakdownList || !info.diagnosis_breakdown) return;
  breakdownList.innerHTML = '';

  for (const [diag, count] of Object.entries(info.diagnosis_breakdown)) {
    const li = document.createElement('li');
    li.style.display = 'flex';
    li.style.justifyContent = 'space-between';
    li.style.padding = '0.2rem 0';
    li.style.borderBottom = '1px solid rgba(255,255,255,0.04)';

    li.innerHTML = `
      <span>${diag}</span>
      <strong style="color: #a5b4fc;">${count} patients</strong>
    `;
    breakdownList.appendChild(li);
  }
}

// ---------------------------------------------------------
// 11. PRESETS, MODAL & TAB HANDLERS
// ---------------------------------------------------------
async function loadPresets() {
  try {
    const res = await fetch('/api/presets');
    const presets = await res.json();

    presetSelect.addEventListener('change', (e) => {
      const presetId = e.target.value;
      if (!presetId || !presets[presetId]) return;

      const pVals = presets[presetId].values;
      for (const [key, val] of Object.entries(pVals)) {
        state.patientData[key] = val;
        const cleanId = key.replace(/[^a-zA-Z0-9]/g, '_');
        const slider = document.getElementById(`slider_${cleanId}`);
        const input = document.getElementById(`input_${cleanId}`);
        if (slider) slider.value = val;
        if (input) input.value = val;
        const meta = state.modelInfo?.features[key];
        if (meta) updateInputRangeBadge(key, val, meta.normal_range);
      }
      updateCalculatedMentzer();
      runInference();
    });

    btnRandomCohort.addEventListener('click', () => {
      if (!state.cohortData.length) return;
      const randIdx = Math.floor(Math.random() * state.cohortData.length);
      loadPatientFromRow(state.cohortData[randIdx]);
    });

  } catch (err) {
    console.error('Presets error:', err);
  }
}

function setupTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const targetId = tab.getAttribute('data-tab');
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      document.getElementById(targetId).classList.add('active');

      if (targetId === 'tab-boundary') {
        setTimeout(() => {
          canvas.width = canvasWrapper.clientWidth;
          canvas.height = canvasWrapper.clientHeight;
          if (state.boundaryData) drawBoundaryCanvas(state.boundaryData);
        }, 50);
      }
    });
  });

  btnResetInputs.addEventListener('click', () => {
    if (state.modelInfo && state.modelInfo.features) {
      buildInputControls(state.modelInfo.features);
      runInference();
    }
  });

  btnRunInference.addEventListener('click', () => {
    runInference();
  });

  document.getElementById('filterCohort').addEventListener('change', (e) => {
    loadCohort(e.target.value);
  });

  btnExportReport.addEventListener('click', () => {
    openReportModal();
  });
  btnCloseModal.addEventListener('click', () => {
    reportModal.classList.remove('active');
  });
  window.addEventListener('click', (e) => {
    if (e.target === reportModal) reportModal.classList.remove('active');
  });
}

function openReportModal() {
  if (!state.latestPrediction) return;

  const pred = state.latestPrediction;
  document.getElementById('reportDate').textContent = new Date().toLocaleString();
  document.getElementById('reportDiagnosis').textContent = pred.diagnosis;
  document.getElementById('reportConfidence').textContent = `Model Confidence: ${pred.confidence}% | Log-Odds Margin (z): ${pred.decision_margin >= 0 ? '+' : ''}${pred.decision_margin}`;

  const tbody = document.getElementById('reportTableBody');
  tbody.innerHTML = '';

  pred.shap_analysis.waterfall.forEach(item => {
    const tr = document.createElement('tr');
    const isBtt = item.direction === 'btt';
    const color = isBtt ? '#c084fc' : '#38bdf8';
    tr.innerHTML = `
      <td>${item.label}</td>
      <td><strong>${item.patient_value} ${item.unit}</strong></td>
      <td>${state.modelInfo.features[item.feature]?.normal_range ? state.modelInfo.features[item.feature].normal_range.join(' - ') : '—'}</td>
      <td style="color: ${color}; font-weight: 700;">${isBtt ? '+' : ''}${item.log_odds_impact}</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById('reportMentzerText').textContent = pred.mentzer_comparison.explanation;
  reportModal.classList.add('active');
}

// Start application
document.addEventListener('DOMContentLoaded', initApp);
