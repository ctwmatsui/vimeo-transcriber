import os
import subprocess
import tempfile
import json
import re
import whisper
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Whisperモデル（初回起動時にダウンロード）
model = None

def get_model():
    global model
    if model is None:
        model = whisper.load_model("medium")
    return model


def download_audio(vimeo_url):
    """yt-dlpでVimeo動画から音声を抽出"""
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "audio.mp3")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "5",
        "--concurrent-fragments", "8",
        "-o", output_path,
        "--no-check-certificates",
        vimeo_url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise Exception(f"ダウンロード失敗: {result.stderr}")

    # yt-dlpが拡張子を変える場合がある
    if not os.path.exists(output_path):
        # .mp3.mp3 等のパターンを探す
        for f in os.listdir(tmp_dir):
            if f.endswith(".mp3"):
                output_path = os.path.join(tmp_dir, f)
                break

    if not os.path.exists(output_path):
        raise Exception("音声ファイルが見つかりません")

    return output_path


def format_timestamp(seconds):
    """秒数をHH:MM:SS形式に変換"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def transcribe_audio(audio_path, with_timestamps=False):
    """Whisperで文字起こし"""
    m = get_model()
    result = m.transcribe(audio_path, language="ja")

    if with_timestamps:
        lines = []
        for seg in result["segments"]:
            ts = format_timestamp(seg["start"])
            lines.append(f"[{ts}] {seg['text'].strip()}")
        return "\n".join(lines)

    return result["text"]


def summarize_with_ollama(transcript, with_timestamps=False):
    """Ollamaのgemma3で議事録要約"""
    timestamp_instruction = ""
    if with_timestamps:
        timestamp_instruction = "- 各ポイントに該当する時間（タイムスタンプ）を付記してください\n"

    prompt = f"""以下は動画の文字起こしテキストです。これを議事録形式で要点をまとめてください。

## フォーマット:
- **概要**: 1〜2文で全体の内容を要約
- **主要なポイント**: 箇条書きで要点を整理
{timestamp_instruction}- **決定事項・アクションアイテム**: あれば記載（なければ省略）
- **補足・メモ**: 重要な補足情報があれば記載（なければ省略）

## 文字起こしテキスト:
{transcript}

## 議事録:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3",
            "prompt": prompt,
            "stream": False,
        },
        timeout=600,
    )

    if response.status_code != 200:
        raise Exception(f"Ollama エラー: {response.text}")

    return response.json()["response"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    url = data.get("url", "").strip()
    with_timestamps = data.get("timestamps", False)

    if not url:
        return jsonify({"error": "URLを入力してください"}), 400

    # Vimeo URLの簡易チェック
    if not re.search(r"vimeo\.com/", url):
        return jsonify({"error": "VimeoのURLを入力してください"}), 400

    try:
        # Step 1: 音声ダウンロード
        audio_path = download_audio(url)

        # Step 2: 文字起こし
        transcript = transcribe_audio(audio_path, with_timestamps)

        # Step 3: 議事録作成
        summary = summarize_with_ollama(transcript, with_timestamps)

        # 一時ファイル削除
        try:
            os.remove(audio_path)
            os.rmdir(os.path.dirname(audio_path))
        except:
            pass

        return jsonify({
            "transcript": transcript,
            "summary": summary,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5555)
