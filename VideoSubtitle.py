import os
import ffmpeg
import whisper
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

class VideoSubtitle:
    def __init__(self):
        self.drive_service = self.authenticate_google_drive()
        self.model = whisper.load_model("base")  # Load the Whisper model

    def authenticate_google_drive(self):
        """Authenticate and create a Google Drive API service."""
        SCOPES = ['https://www.googleapis.com/auth/drive']
        credentials_path = '/content/drive/My Drive/SD-Data/comfyui-n8n-aici01-7679b55c962b.json'
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES)
        return build('drive', 'v3', credentials=credentials)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": "https://example.com/video.mp4", "tooltip": "The direct URL of the video."}),
                "fontname": ("STRING", {"default": "Arial", "tooltip": "Font name for subtitles."}),
                "fontsize": ("INT", {"default": 20, "min": 10, "max": 100, "step": 1, "tooltip": "Font size for subtitles."}),
                "primary_color": ("STRING", {"default": "white", "tooltip": "Primary subtitle color."}),
                "alignment": ("INT", {"default": 2, "min": 0, "max": 9, "tooltip": "Subtitle alignment (e.g. 2 for center, 7 for bottom)."})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "process_video"
    CATEGORY = "video"

    def download_video(self, video_url):
        """Download video from URL."""
        output_path = "/tmp/input_video.mp4"
        os.system(f"wget -O {output_path} {video_url}")
        return output_path

    def extract_audio(self, video_path):
        """Extract audio from video using ffmpeg."""
        audio_path = "/tmp/audio.wav"
        ffmpeg.input(video_path).output(audio_path).run()
        return audio_path

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper model."""
        result = self.model.transcribe(audio_path)
        return result["text"], result["segments"]

    def generate_subtitles(self, segments, output_path, fontname, fontsize, primary_color, alignment):
        """Generate subtitle file in ASS format using transcribed segments."""
        subtitle_path = f"{output_path}.ass"
        with open(subtitle_path, "w") as f:
            f.write(f"""[Script Info]
Title: Subtitles
ScriptType: v4.00+
PlayDepth: 0

[V4+ Styles]
Style: Default,{fontname},{fontsize},{primary_color},&H000000FF,&H00000000,-1,0,0,0,100,100,0,2,{alignment},20,20,20,1

[Events]
Format: Layer, Start, End, Style, Text
""")
            for segment in segments:
                start_time = self.convert_time(segment["start"])
                end_time = self.convert_time(segment["end"])
                text = segment["text"]
                f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")
        return subtitle_path

    def convert_time(self, seconds):
        """Convert time in seconds to ASS format (H:MM:SS.CS)."""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes):02}:{int(seconds):05.2f}"

    def add_subtitles_to_video(self, video_path, subtitle_path):
        """Add subtitles to video using ffmpeg."""
        output_video_path = "/tmp/output_video.mp4"
        ffmpeg.input(video_path).output(output_video_path, vf=f"ass={subtitle_path}").run()
        return output_video_path

    def upload_to_google_drive(self, video_path):
        """Upload video to Google Drive and return the shared URL."""
        try:
            file_metadata = {'name': os.path.basename(video_path), 'parents': ['1fZyeDT_eW6ozYXhqi_qLVy-Xnu5JD67a']}
            media = MediaFileUpload(video_path, mimetype='video/mp4')
            file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

            file_id = file.get('id')
            self.drive_service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()
            return f"https://drive.google.com/uc?id={file_id}"
        except Exception as e:
            print(f"An error occurred while uploading to Google Drive: {e}")
            return None

    def process_video(self, video_url, fontname="Arial", fontsize=20, primary_color="white", alignment=2):
        """Main function to process video and generate subtitles."""
        # Step 1: Download video
        video_path = self.download_video(video_url)

        # Step 2: Extract audio from video
        audio_path = self.extract_audio(video_path)

        # Step 3: Transcribe audio to text using Whisper
        transcript, segments = self.transcribe_audio(audio_path)

        # Step 4: Generate subtitle file
        subtitle_path = self.generate_subtitles(segments, "/tmp/subtitles", fontname, fontsize, primary_color, alignment)

        # Step 5: Add subtitles to video
        output_video_path = self.add_subtitles_to_video(video_path, subtitle_path)

        # Step 6: Upload video to Google Drive and get public URL
        public_url = self.upload_to_google_drive(output_video_path)

        return (public_url,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "VideoSubtitle": VideoSubtitle
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSubtitle": "Video Subtitle Node"
}
