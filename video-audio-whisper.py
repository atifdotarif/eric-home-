from xml.parsers.expat import model
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import RequestEntityTooLarge
import whisper
from supabase import create_client, Client
import tempfile
import cv2
import traceback
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU-only

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

# ---------------- Supabase Connection ----------------
SUPABASE_URL = "https://fmfanuooupxhecbmkkjp.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_8ciah4qOvKKdACDVMUe4Yw_q8gwSAGN"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------- Load Whisper Model ----------------
# print("Loading Whisper model...")
# # WHISPER_MODEL = whisper.load_model("base")
# WHISPER_MODEL = whisper.load_model("tiny", device="cpu")
# print("Whisper model loaded.")

# ---------------- Helper: Format Timestamp ----------------
def format_timestamp(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    return f"{h:02}:{m:02}:{s:02},{millis:03}"

# ---------------- Whisper Transcription ----------------
def transcribe_video(file_bytes):
    if not file_bytes:
        raise ValueError("Uploaded video/audio file is empty")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

    try:
        # Use try-except to catch empty/invalid audio
        # result = WHISPER_MODEL.transcribe(tmp_path)
        model = get_whisper_model()
        result = model.transcribe(tmp_path)
        if "segments" not in result or not result["segments"]:
            raise ValueError("No audio detected in the video.")
        
        transcript_lines = []
        for segment in result.get("segments", []):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            transcript_lines.append(f"[{start} --> {end}] {text}")
        return "\n".join(transcript_lines)
    finally:
        # Always remove temp file
        os.remove(tmp_path)

# ---------------- Frame Processing + Upload ----------------
def process_frames_and_upload(file_bytes, transcript_id):
    if not file_bytes:
        raise ValueError("Empty video cannot be processed for frames")

    frames_metadata = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25  # default fallback

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)

        for sec in range(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Warning: Could not read frame at {sec}s")
                continue

            frame_ts = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                print(f"Warning: Could not encode frame at {sec}s")
                continue

            image_bytes = buffer.tobytes()
            file_path = f"{transcript_id}/frame_{sec}.jpg"

            try:
                supabase.storage.from_("Ericc_video_frames").upload(
                    file_path, image_bytes, {"content-type": "image/jpeg"}
                )
                public_url = supabase.storage.from_("Ericc_video_frames").get_public_url(file_path)
            except Exception as e:
                print(f"Error uploading/getting frame {sec}: {e}")
                public_url = None

            frames_metadata.append({
                "transcript_id": transcript_id,
                "frame_timestamp": frame_ts,
                "frame_storage_url": public_url
            })

        cap.release()
    finally:
        os.remove(tmp_path)

    return frames_metadata

# ---------------- Upload Route ----------------
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        # ---------------- Read form ----------------
        name = request.form.get("name")
        phone_str = request.form.get("phone")

        if not name or not phone_str:
            return jsonify({"error": "Name and phone are required"}), 400

        try:
            phone_num = str(phone_str)
        except ValueError:
            return jsonify({"error": "Invalid phone number"}), 400

        file = request.files.get("video")
        if not file:
            return jsonify({"error": "No video file provided"}), 400

        # file_bytes = file.read()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            file.save(tmp)
            tmp_path = tmp.name
        if not tmp_path or os.path.getsize(tmp_path) == 0:
            return jsonify({"error": "Uploaded video is empty"}), 400

        # ---------------- Transcription ----------------
        try:
            transcript_text = transcribe_video(tmp_path)
        except Exception as e:
            print("Whisper transcription error:\n", traceback.format_exc())
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

        # ---------------- Insert transcript ----------------
        transcript_payload = {
            "name": name,
            "phoneNumber": phone_num,
            "transcript": transcript_text
        }

        transcript_resp = supabase.table("transcript").insert(transcript_payload).execute()
        if not transcript_resp.data:
            print("Transcript insert failed:", transcript_resp)
            return jsonify({"error": "Failed to save transcript"}), 500

        transcript_id = transcript_resp.data[0]["id"]

        # ---------------- Process frames ----------------
        try:
            frames = process_frames_and_upload(tmp_path, transcript_id)
        except Exception as e:
            print("Frame processing/upload error:\n", traceback.format_exc())
            frames = []  # Don't block upload if frames fail

        # ---------------- Insert frame metadata ----------------
        if frames:
            try:
                frame_resp = supabase.table("frame").insert(frames).execute()
                if not frame_resp.data:
                    print("Frame insert failed:", frame_resp)
            except Exception as e:
                print("Supabase frame insert exception:\n", traceback.format_exc())

        return jsonify({
            "message": "Video processed successfully",
            "transcript_id": transcript_id,
            "frame_count": len(frames)
        })

    except Exception as e:
        print("Unexpected exception:\n", traceback.format_exc())
        return jsonify({"error": f"Upload or transcription failed: {str(e)}"}), 500

# ---------------- Handle Large Uploads ----------------
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"error": "File too large. Max 15 MB allowed."}), 413

# ---------------- Serve HTML ----------------
@app.route("/")
def index():
    return render_template("main.html")

# ---------------- Run App ----------------
# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
