from ultralytics import YOLO
import cv2
import time
import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile

# Pastikan Pydub tahu lokasi ffmpeg
AudioSegment.converter = r"C:\Users\lapt1\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

def load_labels(label_path):
    """Load labels dari file labels.txt."""
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at: {label_path}")
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def speak_gtts(text):
    """Mengucapkan instruksi dengan gTTS dan pydub."""
    try:
        print(f"[INFO] Mengucapkan: {text}")
        tts = gTTS(text=text, lang='id')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            print(f"[INFO] File suara disimpan: {fp.name}")
            audio = AudioSegment.from_mp3(fp.name)
            play(audio)
            print("[INFO] Suara selesai diputar.")
    except Exception as e:
        print(f"[ERROR] Gagal memutar suara: {e}")

def detect_and_speak(model_path=r'C:\Users\lapt1\Downloads\Tunanetra\best.pt', 
                     label_path=r'C:\Users\lapt1\Downloads\Tunanetra\labels.txt', 
                     conf_threshold=0.5):
    """Deteksi guiding block dengan output suara menggunakan kamera."""
    labels = load_labels(label_path)
    print(f"[INFO] Label dimuat: {labels}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file tidak ditemukan di: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Gagal membuka kamera. Pastikan kamera terhubung dan aktif.")
    
    last_speech_time = 0
    speech_interval = 5  # detik antara dua suara

    print("[INFO] Mulai deteksi, tekan 'q' untuk keluar.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Gagal membaca frame dari kamera.")
            break
        
        results = model(frame, conf=conf_threshold)
        detections_found = False

        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            
            detections_found = True
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                cls = int(box.cls.item())
                label = labels[cls] if cls < len(labels) else f"Unknown_{cls}"
                
                print(f"Deteksi: {label}, Confidence: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                if conf >= conf_threshold:
                    text = f"Belok {label}" if label in ['kanan', 'kiri'] else "Lurus"
                    current_time = time.time()
                    if current_time - last_speech_time >= speech_interval or last_speech_time == 0:
                        print(f"Kondisi suara: Label={label}, Conf={conf:.2f}, Time since last={current_time - last_speech_time:.2f}")
                        print(f"[INFO] Memutar suara: {text}")
                        speak_gtts(text)
                        last_speech_time = current_time
        
        cv2.imshow('Guiding Block Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if detections_found:
            cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program selesai, kamera ditutup.")

if __name__ == "__main__":
    print("[INFO] Memulai deteksi real-time dengan YOLO dan gTTS...")
    try:
        detect_and_speak(conf_threshold=0.5)
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan: {e}")
