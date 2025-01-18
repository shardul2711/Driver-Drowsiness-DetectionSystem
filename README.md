# Driver Drowsiness Detection

This project uses a pre-trained model from Hugging Face to detect drowsiness in real-time using a webcam. The model classifies whether the driver is **awake** or **drowsy**, helping to prevent accidents caused by driver fatigue. 
Additionally, it integrates YOLOv5 to detect whether a phone is being used by the driver and plays a sound when eyes are closed for 1 second or if a phone is detected for 1 second.

## Features
- **Real-time drowsiness detection**: Uses OpenCV to capture video frames from a webcam.
- **Driver monitoring**: Classifies each frame using a Hugging Face model.
- **Phone detection**: Uses YOLOv5 to detect if a phone is being used by the driver.
- **Sound alert**: Plays a sound when eyes are closed for 1 second or when a phone is detected for 1 second.
- **On-screen results**: Displays the classification label and confidence score.
- **Exit mechanism**: Press `q` to quit the video feed.

## Requirements

- Python 3.9 or higher
- Libraries:
  - `transformers`
  - `torch`
  - `opencv-python`
  - `yolov5`
  - `playsound`

## Setup

Follow these steps to set up and run the project locally.

### 1. Clone the repository

Clone the repository to your local machine:
```bash
git clone https://github.com/shardul2711/Driver-Drowsiness-DetectionSystem
cd Driver-Drowsiness-DetectionSystem
```
### 2. Install the dependecies

Clone the repository to your local machine:
```bash
pip install torch torchvision transformers pygame opencv-python
```
### 3. Run the code
