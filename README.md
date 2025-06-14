# 😊 Real-Time Emotion Detection

A real-time emotion recognition system using a Convolutional Neural Network (CNN) trained on the FER2013 dataset. Detects 7 emotions live from a webcam feed using OpenCV for face detection.

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **66.59%** |
| Best Epoch | 46 |
| Dataset | FER2013 (35,000+ images) |
| Emotions | 7 classes |

## Per-Class Performance

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| 😊 Happy | 0.87 | 0.84 | 0.86 |
| 😲 Surprise | 0.76 | 0.83 | 0.80 |
| 🤢 Disgust | 0.75 | 0.56 | 0.64 |
| 😠 Angry | 0.57 | 0.59 | 0.58 |
| 😐 Neutral | 0.62 | 0.63 | 0.62 |
| 😢 Sad | 0.53 | 0.56 | 0.55 |
| 😨 Fear | 0.54 | 0.48 | 0.51 |

## Features

- Real-time emotion detection from webcam
- Detects and tracks faces automatically using OpenCV
- Supports 7 primary emotions
- Runs entirely on local hardware — no cloud or API required
- Trained on FER2013 (35,000+ labelled facial images)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Model Type | Convolutional Neural Network (CNN) |
| Dataset | FER2013 (Kaggle) |
| Language | Python |

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/manny2341/Emotion-Detection.git
cd Emotion-Detection
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the FER2013 dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013) and place it in the project folder.

**5. Process the data**
```bash
python3 data_processing.py
```

**6. Train the model**
```bash
python3 train_model.py
```

**7. Run live emotion detection**
```bash
python3 emotion_detection.py
```

Press **Q** to quit the webcam window.

## Project Structure

```
Emotion-Detection/
├── data_processing.py        # Prepares and processes FER2013 dataset
├── train_model.py            # Builds and trains the CNN model
├── emotion_detection.py      # Live webcam emotion detection
├── Emotion_Detection.h5      # Saved trained model
├── training_history.png      # Accuracy and loss curves
├── haarcascade_frontalface_default.xml  # OpenCV face detector
└── requirements.txt
```

## How It Works

1. OpenCV captures video from the webcam frame by frame
2. Haar Cascade detects face regions in each frame
3. Detected face is resized to 48×48 pixels and converted to grayscale
4. CNN predicts the emotion from the face image
5. Emotion label and confidence are displayed on the video feed in real time

## Author

[@manny2341](https://github.com/manny2341)
