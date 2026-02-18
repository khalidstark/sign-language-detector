import * as tf from '@tensorflow/tfjs';

// MediaPipe is loaded via CDN (global objects)
const { Hands, HAND_CONNECTIONS } = window;
const { Camera } = window;
const { drawConnectors, drawLandmarks } = window;

// Arabic letter mapping - EXACT same order as Python
const LABELS = [
  'ain', 'al', 'aleff', 'bb', 'dal', 'dha',
  'dhad', 'fa', 'gaaf', 'ghain', 'ha', 'haa',
  'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem',
  'nun', 'ra', 'saad', 'seen', 'sheen', 'ta',
  'taa', 'thaa', 'thal', 'toot', 'waw', 'ya',
  'yaa', 'zay'
];

const ENGLISH_TO_ARABIC = {
  'ain': 'ع', 'al': 'ال', 'aleff': 'أ', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ',
  'dhad': 'ض', 'fa': 'ف', 'gaaf': 'ق', 'ghain': 'غ', 'ha': 'هـ', 'haa': 'ح',
  'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 'laam': 'ل', 'meem': 'م',
  'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'ta': 'ط',
  'taa': 'ت', 'thaa': 'ث', 'thal': 'ذ', 'toot': 'ة', 'waw': 'و', 'ya': 'ئ',
  'yaa': 'ي', 'zay': 'ز'
};

// Confidence threshold - same as Python (90%)
const CONFIDENCE_THRESHOLD = 0.90;

// Debounce settings for adding letters
const LETTER_HOLD_TIME = 800; // ms to hold same letter before adding
const COOLDOWN_TIME = 500; // ms cooldown after adding a letter

// State
let model = null;
let camera = null;
let hands = null;
let currentFacingMode = 'user'; // 'user' = front camera, 'environment' = back camera

// Letter tracking state
let lastPredictedLetter = null;
let letterStartTime = null;
let lastAddedTime = 0;

// DOM Elements
const loadingScreen = document.getElementById('loading-screen');
const mainApp = document.getElementById('main-app');
const loadingStatus = document.getElementById('loading-status');
const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const predictedLetterEl = document.getElementById('predicted-letter');
const confidenceFillEl = document.getElementById('confidence-fill');
const confidenceTextEl = document.getElementById('confidence-text');
const sentenceEl = document.getElementById('sentence');
const clearBtn = document.getElementById('clear-btn');
const spaceBtn = document.getElementById('space-btn');
const copyBtn = document.getElementById('copy-btn');
const switchCameraBtn = document.getElementById('switch-camera');

// Build the model architecture in JavaScript (same as Python model)
function buildModel() {
  const model = tf.sequential();

  // Dense layer 1: 63 -> 512, ReLU
  model.add(tf.layers.dense({
    inputShape: [63],
    units: 512,
    activation: 'relu',
    name: 'dense'
  }));

  // BatchNormalization 1
  model.add(tf.layers.batchNormalization({
    name: 'batch_normalization'
  }));

  // Dropout 1 (not used during inference, but needed for architecture)
  model.add(tf.layers.dropout({
    rate: 0.3,
    name: 'dropout'
  }));

  // Dense layer 2: 512 -> 256, ReLU
  model.add(tf.layers.dense({
    units: 256,
    activation: 'relu',
    name: 'dense_1'
  }));

  // BatchNormalization 2
  model.add(tf.layers.batchNormalization({
    name: 'batch_normalization_1'
  }));

  // Dropout 2
  model.add(tf.layers.dropout({
    rate: 0.3,
    name: 'dropout_1'
  }));

  // Dense layer 3: 256 -> 128, ReLU
  model.add(tf.layers.dense({
    units: 128,
    activation: 'relu',
    name: 'dense_2'
  }));

  // Output layer: 128 -> 32, Softmax
  model.add(tf.layers.dense({
    units: 32,
    activation: 'softmax',
    name: 'dense_3'
  }));

  return model;
}

// Load weights from binary file
async function loadWeights(model) {
  const response = await fetch('/model/group1-shard1of1.bin');
  const buffer = await response.arrayBuffer();
  const weights = new Float32Array(buffer);

  // Weight shapes (same order as saved)
  const weightSpecs = [
    { name: 'dense/kernel', shape: [63, 512] },
    { name: 'dense/bias', shape: [512] },
    { name: 'batch_normalization/gamma', shape: [512] },
    { name: 'batch_normalization/beta', shape: [512] },
    { name: 'batch_normalization/moving_mean', shape: [512] },
    { name: 'batch_normalization/moving_variance', shape: [512] },
    { name: 'dense_1/kernel', shape: [512, 256] },
    { name: 'dense_1/bias', shape: [256] },
    { name: 'batch_normalization_1/gamma', shape: [256] },
    { name: 'batch_normalization_1/beta', shape: [256] },
    { name: 'batch_normalization_1/moving_mean', shape: [256] },
    { name: 'batch_normalization_1/moving_variance', shape: [256] },
    { name: 'dense_2/kernel', shape: [256, 128] },
    { name: 'dense_2/bias', shape: [128] },
    { name: 'dense_3/kernel', shape: [128, 32] },
    { name: 'dense_3/bias', shape: [32] }
  ];

  // Extract weights from buffer
  let offset = 0;
  const weightTensors = [];

  for (const spec of weightSpecs) {
    const size = spec.shape.reduce((a, b) => a * b, 1);
    const data = weights.slice(offset, offset + size);
    const tensor = tf.tensor(Array.from(data), spec.shape);
    weightTensors.push(tensor);
    offset += size;
  }

  // Set weights to model layers
  // Dense 1
  model.getLayer('dense').setWeights([weightTensors[0], weightTensors[1]]);
  // BatchNorm 1
  model.getLayer('batch_normalization').setWeights([
    weightTensors[2], weightTensors[3], weightTensors[4], weightTensors[5]
  ]);
  // Dense 2
  model.getLayer('dense_1').setWeights([weightTensors[6], weightTensors[7]]);
  // BatchNorm 2
  model.getLayer('batch_normalization_1').setWeights([
    weightTensors[8], weightTensors[9], weightTensors[10], weightTensors[11]
  ]);
  // Dense 3
  model.getLayer('dense_2').setWeights([weightTensors[12], weightTensors[13]]);
  // Output
  model.getLayer('dense_3').setWeights([weightTensors[14], weightTensors[15]]);

  // Clean up tensors
  weightTensors.forEach(t => t.dispose());

  return model;
}

// Initialize the app
async function init() {
  try {
    // Build and load model
    loadingStatus.textContent = 'تحميل نموذج الذكاء الاصطناعي...';

    model = buildModel();
    console.log('Model architecture built');

    await loadWeights(model);
    console.log('Model weights loaded');

    // Warm up the model with a dummy prediction
    const warmupResult = model.predict(tf.zeros([1, 63]));
    warmupResult.dispose();
    console.log('Model warmed up');

    // Initialize MediaPipe Hands
    loadingStatus.textContent = 'تهيئة كاشف اليد...';

    hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5
    });

    hands.onResults(onResults);

    // Initialize camera
    loadingStatus.textContent = 'تشغيل الكاميرا...';
    await startCamera();

    // Show main app
    loadingScreen.classList.add('hidden');
    mainApp.classList.remove('hidden');

    // Setup event listeners
    setupEventListeners();

  } catch (error) {
    console.error('Initialization error:', error);
    loadingStatus.textContent = 'خطأ: ' + error.message;
  }
}

// Start camera with specified facing mode
async function startCamera() {
  if (camera) {
    camera.stop();
  }

  camera = new Camera(videoElement, {
    onFrame: async () => {
      await hands.send({ image: videoElement });
    },
    facingMode: currentFacingMode,
    width: 640,
    height: 480
  });

  await camera.start();
}

// Process MediaPipe results
function onResults(results) {
  const canvasCtx = canvasElement.getContext('2d');

  // Set canvas size to match video
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;

  // Clear canvas
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    // Draw hand landmarks on canvas
    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
      color: '#6366f1',
      lineWidth: 2
    });
    drawLandmarks(canvasCtx, landmarks, {
      color: '#22c55e',
      lineWidth: 1,
      radius: 3
    });

    // Extract landmarks for prediction - EXACT same as Python
    // Python: landmarks.append([landmark.x, landmark.y, landmark.z])
    // Then: np.array(hand_landmarks).flatten().reshape(1, -1)
    const flatLandmarks = [];
    for (const landmark of landmarks) {
      flatLandmarks.push(landmark.x, landmark.y, landmark.z);
    }

    // Make prediction
    predictLetter(flatLandmarks);
  } else {
    // No hand detected
    updatePrediction(null, 0);
  }
}

// Predict letter from landmarks
function predictLetter(landmarks) {
  if (!model || landmarks.length !== 63) {
    updatePrediction(null, 0);
    return;
  }

  // Create tensor - shape [1, 63]
  const inputTensor = tf.tensor2d([landmarks], [1, 63]);

  // Get prediction
  const prediction = model.predict(inputTensor);
  const probabilities = prediction.dataSync();

  // Find max probability and class
  let maxProb = 0;
  let maxClass = 0;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxClass = i;
    }
  }

  // Clean up tensors
  inputTensor.dispose();
  prediction.dispose();

  // Apply confidence threshold - same as Python (90%)
  if (maxProb > CONFIDENCE_THRESHOLD) {
    const englishLabel = LABELS[maxClass];
    const arabicLetter = ENGLISH_TO_ARABIC[englishLabel];
    updatePrediction(arabicLetter, maxProb);
    handleLetterDetection(arabicLetter);
  } else {
    updatePrediction(null, maxProb);
    resetLetterTracking();
  }
}

// Update the prediction display
function updatePrediction(letter, confidence) {
  const confidencePercent = Math.round(confidence * 100);

  if (letter) {
    predictedLetterEl.textContent = letter;
    predictedLetterEl.className = 'predicted-letter high-confidence';
    confidenceFillEl.className = 'confidence-fill high';
  } else {
    predictedLetterEl.textContent = confidencePercent > 0 ? '?' : '-';
    predictedLetterEl.className = 'predicted-letter low-confidence';
    confidenceFillEl.className = 'confidence-fill low';
  }

  confidenceFillEl.style.width = `${confidencePercent}%`;
  confidenceTextEl.textContent = `${confidencePercent}%`;
}

// Handle letter detection for sentence building
function handleLetterDetection(letter) {
  const now = Date.now();

  // Check cooldown
  if (now - lastAddedTime < COOLDOWN_TIME) {
    return;
  }

  if (letter === lastPredictedLetter) {
    // Same letter - check if held long enough
    if (letterStartTime && (now - letterStartTime >= LETTER_HOLD_TIME)) {
      addLetterToSentence(letter);
      lastAddedTime = now;
      letterStartTime = now; // Reset for potential repeat
    }
  } else {
    // New letter - start tracking
    lastPredictedLetter = letter;
    letterStartTime = now;
  }
}

// Reset letter tracking
function resetLetterTracking() {
  lastPredictedLetter = null;
  letterStartTime = null;
}

// Add letter to sentence
function addLetterToSentence(letter) {
  sentenceEl.textContent += letter;

  // Vibrate on mobile (if supported)
  if (navigator.vibrate) {
    navigator.vibrate(50);
  }
}

// Setup event listeners
function setupEventListeners() {
  // Clear button
  clearBtn.addEventListener('click', () => {
    sentenceEl.textContent = '';
    showToast('تم المسح');
  });

  // Space button
  spaceBtn.addEventListener('click', () => {
    sentenceEl.textContent += ' ';
  });

  // Copy button
  copyBtn.addEventListener('click', async () => {
    const text = sentenceEl.textContent;
    if (text) {
      try {
        await navigator.clipboard.writeText(text);
        showToast('تم النسخ!');
      } catch (err) {
        showToast('فشل النسخ');
      }
    }
  });

  // Switch camera button
  switchCameraBtn.addEventListener('click', async () => {
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    await startCamera();
    showToast(currentFacingMode === 'user' ? 'الكاميرا الأمامية' : 'الكاميرا الخلفية');
  });
}

// Show toast notification
function showToast(message) {
  const existing = document.querySelector('.toast');
  if (existing) {
    existing.remove();
  }

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 2000);
}

// Start the app
init();
