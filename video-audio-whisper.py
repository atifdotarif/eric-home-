from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import RequestEntityTooLarge
import whisper
from supabase import create_client, Client
import tempfile
import cv2
import traceback
import os

# ---------------- Environment ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU-only

# ---------------- Whisper Lazy Load ----------------
WHISPER_MODEL = None

def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("Lazy-loading Whisper tiny model...")
        WHISPER_MODEL = whisper.load_model("tiny", device="cpu")
    return WHISPER_MODEL

# ---------------- Flask App ----------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB max upload

# ---------------- Supabase ----------------
SUPABASE_URL = "https://fmfanuooupxhecbmkkjp.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_8ciah4qOvKKdACDVMUe4Yw_q8gwSAGN"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------- Helpers ----------------
def format_timestamp(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    return f"{h:02}:{m:02}:{s:02},{millis:03}"

# ---------------- Whisper Transcription ----------------
def transcribe_video(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise ValueError("Video file does not exist")

    model = get_whisper_model()
    result = model.transcribe(video_path)

    if not result.get("segments"):
        raise ValueError("No audio detected in the video.")

    transcript_lines = []
    for segment in result["segments"]:
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        transcript_lines.append(f"[{start} --> {end}] {text}")

    return "\n".join(transcript_lines)

# ---------------- Frame Processing ----------------
def process_frames_and_upload(video_path: str, transcript_id: int):
    frames_metadata = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)

    for sec in range(duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_ts = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        image_bytes = buffer.tobytes()
        file_path = f"{transcript_id}/frame_{sec}.jpg"

        try:
            supabase.storage.from_("Ericc_video_frames").upload(
                file_path, image_bytes, {"content-type": "image/jpeg"}
            )
            public_url = supabase.storage.from_("Ericc_video_frames").get_public_url(file_path)
        except Exception:
            public_url = None

        frames_metadata.append({
            "transcript_id": transcript_id,
            "frame_timestamp": frame_ts,
            "frame_storage_url": public_url
        })

    cap.release()
    return frames_metadata

# ---------------- Upload Route ----------------
@app.route("/upload", methods=["POST"])
def upload_file():
    tmp_path = None
    try:
        name = request.form.get("name")
        phone = request.form.get("phone")
        file = request.files.get("video")

        if not name or not phone:
            return jsonify({"error": "Name and phone are required"}), 400
        if not file:
            return jsonify({"error": "No video file provided"}), 400

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        if os.path.getsize(tmp_path) == 0:
            return jsonify({"error": "Uploaded video is empty"}), 400

        transcript_text = transcribe_video(tmp_path)

        transcript_resp = supabase.table("transcript").insert({
            "name": name,
            "phoneNumber": phone,
            "transcript": transcript_text
        }).execute()

        transcript_id = transcript_resp.data[0]["id"]

        frames = process_frames_and_upload(tmp_path, transcript_id)
        if frames:
            supabase.table("frame").insert(frames).execute()

        return jsonify({
            "message": "Video processed successfully",
            "transcript_id": transcript_id,
            "frame_count": len(frames)
        })

    except Exception as e:
        print("Error:\n", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------- Errors ----------------
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"error": "File too large. Max 15 MB allowed."}), 413

# ---------------- Frontend ----------------
@app.route("/")
def index():
    return render_template("main.html")

# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
