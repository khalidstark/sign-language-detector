# Arabic Sign Language Detector

Real-time recognition of the Arabic sign language alphabet (finger spelling) from a webcam feed, translating hand signs into Arabic letters on screen.

## Screenshots

<!-- TODO: add real screenshots/GIF, e.g.
<p align="center">
  <img src="assets/screenshots/desktop-app.png" width="300" />
  <img src="assets/screenshots/web-demo.gif" width="300" />
</p>
-->

---

## How It Works

1. **Hand tracking** — [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) extracts 21 hand landmarks per frame from the camera feed
2. **Classification** — the landmarks are fed into a trained Keras model (`asl_model_improved.h5`) that classifies the hand shape into one of the Arabic sign language letters
3. **Translation** — predicted labels are mapped from their English transliteration (e.g. `jeem`, `laam`, `meem`) to the corresponding Arabic letter (ج, ل, م) and rendered on screen with Arabic-aware text rendering (Pillow)

Two implementations are included:

- **Desktop app** (`final.py`) — Python + OpenCV + Tkinter, runs the full pipeline locally against a webcam
- **Web demo** (`arabic-sign-language-web/`) — the same model converted to TensorFlow.js and run in-browser (Vite-based, deployable to Firebase Hosting)

## Tech Stack

| Layer | Tech |
|---|---|
| Hand landmark detection | MediaPipe Hands |
| Model | TensorFlow / Keras (`.h5`), converted to TensorFlow.js for web |
| Desktop UI | Python, OpenCV, Tkinter, Pillow |
| Web UI | JavaScript, Vite, TensorFlow.js |
| Deployment (web) | Firebase Hosting |

## Setup

### Desktop

```bash
pip install opencv-python mediapipe tensorflow pillow numpy
python final.py
```

Requires a connected webcam. Press the window's close control or `q` to quit.

### Web

```bash
cd arabic-sign-language-web
npm install
npm run dev
```
