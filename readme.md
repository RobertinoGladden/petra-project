# Petra: Smart Guide for the Visually Impaired

**Petra** is an intelligent AI assistant designed to help visually impaired users navigate using a YOLOv8 object detection model, voice guidance with Google Text-to-Speech (gTTS), and video processing overlays.
---
## 🎬 Demo

![Petra AI Demo]([C:\Users\lapt1\Downloads\Tunanetra\petraGray.avi](https://github.com/RobertinoGladden/petra-project/blob/main/petraGray.avi))

## 📁 Project Structure

```bash
├── dataset
│   ├── test
│   ├── train
│   ├── valid
│   ├── data.yaml
│   ├── README.dataset.txt
│   └── README.roboflow.txt
│
├── model
│   ├── best.engine
│   ├── best.onnx
│   ├── best.pt          <- YOLOv8 model used
│   ├── labels.txt        <- Class labels for object detection
│   ├── model.py
│   └── readme.md
│
├── voice.py             <- Generates voice guidance
├── voiceGrayscale.py    <- Experimental grayscale version
├── test.py              <- Test script for the pipeline
├── yolo11n.pt           <- Alternate model checkpoint
└── settings.json        <- Optional configuration settings
```

---

## 🚀 Features

- **Real-time Object Detection** using YOLOv8
- **Voice Guidance** powered by Google TTS
- **Video Overlay** with frame, timestamp, and detected object bounding boxes
- **Bounding Zone Filter**: Only objects inside the defined area are considered for instructions
- **Multiformat Output**: Generates annotated video and overlays audio

---

## 🧠 How It Works

1. Loads a YOLO model and class labels
2. Processes each frame of the input video
3. Checks if detected object lies within a bounding region
4. If true, plays a voice instruction (e.g., "Turn Right")
5. Displays current time, frame number, and FPS on screen
6. Draws bounding boxes and overlays text on video

---

## ⚙️ Dependencies

- Python >= 3.8
- OpenCV
- Ultralytics YOLOv8
- Google Text-to-Speech (gTTS)
- MoviePy
- NumPy, PIL

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔊 Example Labels (labels.txt)

```bash
kiri
kanan
lurus
```

---

## 🎥 Input & Output

- **Input Video:** Defined in `input_video_path`
- **Temporary Output:** Intermediate AVI file
- **Final Output:** MP4 video with overlaid instructions and audio guidance

---

## 🗂 Configuration Tips

- Edit `model_path`, `label_path`, `input_video_path` in the script for custom use
- You can define your own bounding area coordinates inside the code
- Works best with fine-tuned YOLOv8 model on navigation-related classes (left, right, straight)

---

## 🙌 Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Google Text-to-Speech](https://pypi.org/project/gTTS/)
- [MoviePy](https://zulko.github.io/moviepy/)

---

## 💡 Future Improvements

- Real-time webcam inference
- Integration with haptic feedback
- Localization support for multiple languages

---

## 📬 Contact

For questions or collaboration, feel free to reach out!

---

**Project by:** Robertino Gladden Narendra

**Purpose:** Assistive technology for blind/visually impaired navigation

---


> "Guiding with AI. Empowering through vision."
