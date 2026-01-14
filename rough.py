from flask import Flask, request, jsonify
import whisper
from supabase import create_client, Client
import tempfile
import cv2
import numpy as np
import traceback

app = Flask(__name__)

# ---------------- Supabase Connection ----------------
SUPABASE_URL = "https://fmfanuooupxhecbmkkjp.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_8ciah4qOvKKdACDVMUe4Yw_q8gwSAGN"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------- Load Whisper Model Once ----------------
print("Loading Whisper model...")
WHISPER_MODEL = whisper.load_model("base")
print("Whisper model loaded.")

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
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        tmp.flush()

        result = WHISPER_MODEL.transcribe(tmp.name)

    transcript_lines = []
    for segment in result.get("segments", []):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        transcript_lines.append(f"[{start} --> {end}] {text}")

    return "\n".join(transcript_lines)

# ---------------- Frame Processing + Upload ----------------
def process_frames_and_upload(file_bytes, transcript_id):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        tmp.flush()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)

        frames_metadata = []

        for sec in range(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame at {sec}s")
                continue

            frame_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                print(f"Warning: Could not encode frame at {sec}s")
                continue

            image_bytes = buffer.tobytes()
            file_path = f"{transcript_id}/frame_{sec}.jpg"

            # Upload to Supabase storage with upsert=True to avoid conflicts
            try:
                supabase.storage.from_("Ericc_video_frames").upload(
                    file_path,
                    image_bytes,
                    {"content-type": "image/jpeg"}
                )
            except Exception as e:
                print(f"Error uploading frame {sec}: {e}")
                continue

            try:
                public_url = supabase.storage.from_("Ericc_video_frames").get_public_url(file_path)
                print(f"Public URL for frame {sec}: {public_url}")
            except Exception as e:
                print(f"Error getting public URL for frame {sec}: {e}")
                public_url = None

            frames_metadata.append({
                "transcript_id": transcript_id,
                "frame_timestamp": frame_ts,
                "frame_storage_url": public_url
            })

            print(f"Uploaded frame @ {frame_ts:.3f}s")

        cap.release()

    return frames_metadata

# ---------------- Upload Route ----------------
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        name = request.form.get("name")
        phone_str = request.form.get("phone")

        try:
            phone_num = int(phone_str)
        except ValueError:
            return jsonify({"error": "Invalid phone number"}), 400

        file = request.files.get("video")
        if not file:
            return jsonify({"error": "No video file provided"}), 400

        file_bytes = file.read()

        # 1️⃣ Transcribe
        try:
            transcript_text = transcribe_video(file_bytes)
        except Exception as e:
            print("Whisper transcription error:\n", traceback.format_exc())
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

        # 2️⃣ Insert transcript
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

        # 3️⃣ Extract frames & upload
        try:
            frames = process_frames_and_upload(file_bytes, transcript_id)
        except Exception as e:
            print("Frame processing/upload error:\n", traceback.format_exc())
            return jsonify({"error": f"Frame processing failed: {str(e)}"}), 500

        # 4️⃣ Insert frame metadata
        if frames:
            try:
                frame_resp = supabase.table("frame").insert(frames).execute()
                if not frame_resp.data:
                    print("Frame insert failed:", frame_resp)
                    return jsonify({"error": "Failed to save frames"}), 500
            except Exception as e:
                print("Supabase frame insert exception:\n", traceback.format_exc())
                return jsonify({"error": f"Failed to save frames: {str(e)}"}), 500

        return jsonify({
            "message": "Video processed successfully",
            "transcript_id": transcript_id,
            "frame_count": len(frames)
        })

    except Exception as e:
        print("Unexpected exception:\n", traceback.format_exc())
        return jsonify({"error": f"Upload or transcription failed: {str(e)}"}), 500

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(debug=True)
