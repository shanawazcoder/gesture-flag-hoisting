# gesture-flag-hoisting

This project uses **Computer Vision** and **Hand Gesture Recognition** to hoist the Indian flag using a simple upward hand gesture.

The system uses your **webcam** to detect hand landmarks through **MediaPipe**, and when an upward gesture is detected, the **Indian flag rises on the screen with a waving animation**.



# Features

- Real-time **hand gesture detection**
- **Upward gesture triggers flag hoisting**
- Animated **waving Indian flag**
- Works using a **webcam**
- Built using **OpenCV + MediaPipe**

---

# Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Math

---

# Installation

1. Clone the repository

git clone https://github.com/yourusername/gesture-flag-hoisting.git
cd gesture-flag-hoisting

Install required libraries

pip install opencv-python mediapipe numpy
Run the Project
python main.py

After running:

Your webcam will start.

Show your hand upward gesture.

The Indian flag will start hoisting with waving animation.

Press Q to exit.

Gesture Used

The system detects:

Wrist

Middle finger tip

Middle finger joint

If the middle finger points upward relative to the wrist, the gesture is detected and the flag starts hoisting.
