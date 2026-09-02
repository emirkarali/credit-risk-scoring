const form = document.getElementById('riskForm');
const submitButton = document.getElementById('submitButton');
const statusText = document.getElementById('statusText');
const riskBadge = document.getElementById('riskBadge');
const probabilityValue = document.getElementById('probabilityValue');
const thresholdValue = document.getElementById('thresholdValue');
const baseRateValue = document.getElementById('baseRateValue');
const liftValue = document.getElementById('liftValue');
const featureCountValue = document.getElementById('featureCountValue');
const riskDescription = document.getElementById('riskDescription');
const scoreRing = document.getElementById('scoreRing');
const calibrationNote = document.getElementById('calibrationNote');
const totalColumnsText = document.getElementById('totalColumnsText');

const BAND_COLORS = {
  'Low Risk': '#38d39f',
  'Elevated Risk': '#f5b942',
  'High Risk': '#ff6b7d'
};

const BAND_CLASSES = {
  'Low Risk': 'risk-low',
  'Elevated Risk': 'risk-medium',
  'High Risk': 'risk-high'
};

let modelMetadata = null;

// Fields that must reach the API as numbers rather than form strings.
const numericFields = [
  'CNT_CHILDREN',
  'AMT_INCOME_TOTAL',
  'AMT_CREDIT',
  'AMT_ANNUITY',
  'AMT_GOODS_PRICE',
  'REGION_RATING_CLIENT',
  'EXT_SOURCE_1',
  'EXT_SOURCE_2',
  'EXT_SOURCE_3',
  'age_years',
  'years_employed'
];

const categoryFields = [
  'NAME_CONTRACT_TYPE',
  'NAME_INCOME_TYPE',
  'NAME_EDUCATION_TYPE',
  'NAME_FAMILY_STATUS',
  'NAME_HOUSING_TYPE'
];

function sanitizeDecimalInput(event) {
  const input = event.target;
  let value = input.value;

  if (value === '') return;

  value = value.replace(/[^0-9.\-]/g, '');
  const minusMatches = value.match(/-/g) || [];
  if (minusMatches.length > 1) {
    value = value.replace(/-/g, '');
  }

  const dotMatches = value.match(/\./g) || [];
  if (dotMatches.length > 1) {
    const [firstDot] = value.split('.');
    value = `${firstDot}.${value.slice(firstDot.length + 1).replace(/\./g, '')}`;
  }

  if (value.includes('-') && !value.startsWith('-')) {
    value = value.replace(/-/g, '');
  }

  input.value = value;
}

function attachDecimalSanitizers() {
  document
    .querySelectorAll('input[inputmode="decimal"]')
    .forEach((input) => input.addEventListener('input', sanitizeDecimalInput));
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.style.color = isError ? '#ff6b7d' : '#a3b6d6';
}

function toNumber(value) {
  if (value === '' || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

// The API owns the DAYS_BIRTH / DAYS_EMPLOYED conversion and all derived
// ratios, so the browser only cleans up types and drops empty values.
function normalizePayload(raw) {
  const payload = { ...raw };

  for (const field of numericFields) {
    if (field in payload) {
      const parsed = toNumber(payload[field]);
      if (parsed === null) {
        delete payload[field];
      } else {
        payload[field] = parsed;
      }
    }
  }

  if (payload.FLAG_OWN_CAR === 'yes') payload.FLAG_OWN_CAR = 'Y';
  if (payload.FLAG_OWN_CAR === 'no') payload.FLAG_OWN_CAR = 'N';
  if (payload.FLAG_OWN_REALTY === 'yes') payload.FLAG_OWN_REALTY = 'Y';
  if (payload.FLAG_OWN_REALTY === 'no') payload.FLAG_OWN_REALTY = 'N';

  for (const field of [...categoryFields, 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']) {
    if (field in payload && (payload[field] === '' || payload[field] === null)) {
      delete payload[field];
    }
  }

  return payload;
}

async function loadMetadata() {
  try {
    const response = await fetch('/metadata');
    if (!response.ok) throw new Error('metadata unavailable');
    modelMetadata = await response.json();

    if (totalColumnsText) {
      totalColumnsText.textContent = modelMetadata.total_columns;
    }

    const auc = modelMetadata.test_metrics && modelMetadata.test_metrics.roc_auc;
    const aucText = auc ? ` Held-out test ROC-AUC ${auc.toFixed(3)}.` : '';

    calibrationNote.textContent = modelMetadata.calibrated
      ? `Probabilities are isotonically calibrated, so the figure above can be read as an estimated chance of payment difficulty.${aucText}`
      : `This model is NOT calibrated. The figure above is a ranking score, not a probability — do not read it as a chance of default.${aucText}`;
    calibrationNote.className = modelMetadata.calibrated
      ? 'calibration-note'
      : 'calibration-note warning';
  } catch (error) {
    calibrationNote.textContent =
      'Model metadata could not be loaded; treat the score as a ranking value only.';
    calibrationNote.className = 'calibration-note warning';
  }
}

function renderResult(data) {
  const probability = Number(data.probability);
  const band = data.risk_label;
  const scorePercent = Math.max(0, Math.min(100, probability * 100));
  const ringColor = BAND_COLORS[band] || '#ff6b7d';

  riskBadge.textContent = band;
  riskBadge.className = `risk-badge ${BAND_CLASSES[band] || 'risk-high'}`;

  probabilityValue.textContent = `${scorePercent.toFixed(1)}%`;
  thresholdValue.textContent = `${(Number(data.threshold) * 100).toFixed(1)}%`;
  baseRateValue.textContent = `${(Number(data.base_rate) * 100).toFixed(1)}%`;
  liftValue.textContent = `${Number(data.lift_vs_base_rate).toFixed(2)}x`;
  featureCountValue.textContent = `${data.features_supplied} / ${data.total_columns}`;

  const lift = Number(data.lift_vs_base_rate);
  const comparison =
    lift >= 1
      ? `about ${lift.toFixed(1)}x the average applicant`
      : `below the average applicant`;

  const imputedCount = data.total_columns - data.features_supplied;

  riskDescription.textContent =
    `${data.calibrated ? 'Estimated' : 'Scored'} risk of payment difficulty is ${comparison}. ` +
    `${data.is_high_risk ? 'Above' : 'Below'} the ${(data.threshold * 100).toFixed(1)}% decision ` +
    `threshold. ${imputedCount} of ${data.total_columns} columns were imputed from training data.`;

  scoreRing.style.setProperty('--score', scorePercent.toFixed(1));
  scoreRing.style.setProperty('--ring-color', ringColor);
}

function renderError(message) {
  setStatus(message, true);
  riskBadge.textContent = 'Error';
  riskBadge.className = 'risk-badge risk-high';
  probabilityValue.textContent = '--';
  thresholdValue.textContent = '--';
  baseRateValue.textContent = '--';
  liftValue.textContent = '--';
  featureCountValue.textContent = '--';
  riskDescription.textContent = 'Please review the form data and try again.';
  scoreRing.style.setProperty('--score', 0);
  scoreRing.style.setProperty('--ring-color', '#ff6b7d');
}

async function submitForm(event) {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = normalizePayload(Object.fromEntries(formData.entries()));

  if (Object.keys(payload).length === 0) {
    renderError('Please fill in at least one field before scoring.');
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = 'Scoring in progress...';
  setStatus('Processing the submitted data...');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Prediction request failed.');
    }

    renderResult(data);
    setStatus('Prediction completed successfully.');
  } catch (error) {
    renderError(error.message || 'An error occurred.');
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = 'Generate prediction';
  }
}

form.addEventListener('submit', submitForm);
attachDecimalSanitizers();
loadMetadata();
