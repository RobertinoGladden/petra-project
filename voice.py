import cv2
from ultralytics import YOLO
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import tempfile
import os
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# === Konfigurasi ===
model_path = r"C:\\Users\\lapt1\\Downloads\\Tunanetra\\best.pt"
label_path = r"C:\\Users\\lapt1\\Downloads\\Tunanetra\\labels.txt"
input_video_path = r"C:\\Users\\lapt1\\Downloads\\Tunanetra\\input.mp4"
temp_video_path = r"petra.avi"
final_output_path = r"petra.mp4"
confidence_threshold = 0.5
speech_interval = 3
text_display_duration = 2
tts_lang = "id"

# FFMPEG untuk MoviePy
os.environ["IMAGEIO_FFMPEG_EXE"] = r"C:\\Users\\lapt1\\ffmpeg-7.1.1-essentials_build\\bin\\ffmpeg.exe"

# === Fungsi ===
def load_labels(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file tidak ditemukan: {path}")
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def overlay_text(frame, status_text, instruksi_text, width):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font_status = ImageFont.truetype("arial.ttf", 18)
        font_class = ImageFont.truetype("arialbd.ttf", 28)
    except:
        font_status = ImageFont.load_default()
        font_class = ImageFont.load_default()

    if status_text:
        text_bbox = draw.textbbox((0, 0), status_text, font=font_status)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = width - text_width - 10
        draw.text((text_x, 10), status_text, font=font_status, fill=(255, 255, 255))

    if instruksi_text:
        draw.text((10, 30), instruksi_text.upper(), font=font_class, fill=(0, 255, 0))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# === Inisialisasi ===
labels = load_labels(label_path)
model = YOLO(model_path)
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise RuntimeError("Gagal membuka video.")
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Gagal membaca frame pertama.")

height, width = first_frame.shape[:2]

bounding_area = np.array([
    [500, 713],
    [650, 713],
    [630, 532],
    [550, 532]
], np.int32)

bounding_mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(bounding_mask, [bounding_area], 255)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

frame_index = 0
last_speech_time = -speech_interval
audio_segments = []
text_to_show = ""
text_timer = 0

print("[INFO] Memulai pemrosesan frame...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = frame_index / fps

    results = model(frame, conf=confidence_threshold, verbose=False)
    for result in results:
        for box in result.boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            label = labels[cls] if cls < len(labels) else f"Unknown_{cls}"
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            if conf >= confidence_threshold and cv2.pointPolygonTest(bounding_area, center, False) >= 0:
                if current_time - last_speech_time >= speech_interval:
                    instruksi = f"Belok {label}" if label in ['kanan', 'kiri'] else "Straight"
                    print(f"[DETEKSI] {label} @ {current_time:.2f}s → '{instruksi}'")

                    tts = gTTS(text=instruksi, lang=tts_lang)
                    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                    tts.save(temp_audio.name)
                    audio_segments.append((temp_audio.name, current_time))

                    text_to_show = instruksi
                    text_timer = current_time
                    last_speech_time = current_time
                    break
        if text_to_show:
            break

    if text_to_show and (current_time - text_timer <= text_display_duration):
        frame = overlay_text(frame, "RUNNING MODEL", text_to_show, width)
    else:
        text_to_show = ""
        frame = overlay_text(frame, "RUNNING MODEL", "", width)

    # draw_bounding_area(frame, bounding_area, color=(0, 255, 0), thickness=2)

    cv2.putText(frame, f"Time: {current_time:.2f}s / {total_frames / fps:.2f}s", (10, height - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_index} / {total_frames}", (10, height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Petra - Penuntun Tunanetra", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    for result in results:
        for box in result.boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            label = labels[cls] if cls < len(labels) else f"Unknown_{cls}"
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)
    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("[INFO] Video selesai. Menggabungkan audio...")

video_clip = VideoFileClip(temp_video_path)
audio_clips = [AudioFileClip(path).set_start(start) for path, start in audio_segments]
final_video = video_clip.set_audio(CompositeAudioClip(audio_clips)) if audio_clips else video_clip
final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")

print(f"[SELESAI] Video akhir disimpan di: {final_output_path}")

for path, _ in audio_segments:
    if os.path.exists(path):
        os.remove(path)