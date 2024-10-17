import os
import subprocess
import whisper
import torch
import time
import requests
import tempfile
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from moviepy.editor import VideoFileClip
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Timestamped_word class
@dataclass
class Timestamped_word:
    start_time: int  # both start_time and end_time are in ms (So, multiply by 1000 from segments)
    end_time: int
    word: str

    def to_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "word": self.word
        }

# Configurations
base_dir = os.path.dirname(__file__)

config_path = os.path.join(base_dir, 'config.json')
with open(config_path) as f:
    config = json.load(f)

word_options_path = os.path.join(base_dir, 'word_options.json')
with open(word_options_path) as f:
    word_options_list = json.load(f)

RESOURCES_DIR = config['RESOURCES_DIR']
VIDEO_DIR = config['VIDEO_DIR']
AUDIO_DIR = config['AUDIO_DIR']
JSON_DIR = config['JSON_DIR']
SUBTITLES_DIR = config['SUBTITLES_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']
TMP_OUTPUT_DIR = config['TMP_OUTPUT_DIR']
TMP_SUBTITLES_DIR = config['TMP_SUBTITLES_DIR']
LOGS_DIR = config['LOGS_DIR']
FONTS_DIR = config['FONTS_DIR']
THUMBNAILS_DIR = config['THUMBNAILS_DIR']
PARAMS_JSON_1_PATH = config['PARAMS_JSON_1_PATH']
PARAMS_JSON_2_PATH = config['PARAMS_JSON_2_PATH']
FONTS_JSON_PATH = config['FONTS_JSON_PATH']

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(SUBTITLES_DIR, exist_ok=True)

word_options_index_map = {
    "1-2 words per line": "1",
    "3-4 words per line": "2",
    "5-7 words per line": "3",
    "8-10 words per line": "4",
    "11-12 words per line": "5",
    "13-15 words per line": "6",
}

video_quality_map = {  # maps video quality to crf value that varies from 0 to 51, lower the better quality
    'highest': '12',
    'high': '15',
    'medium': '18',
    'low': '20',
    'lowest': '23',
    'reduced': '30'
}

# Utility functions
def json_read(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def json_write(json_path, text):
    with open(json_path, 'w') as file:
        json.dump(text, file, indent=4)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_text_file(file_path, text):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def generate_log_name():
    current_datetime = datetime.now()
    date_string = current_datetime.strftime('%Y-%m-%d')
    time_string = current_datetime.strftime('%H%M%S')
    return f"{date_string}_{time_string}"

def get_curr_log_file_path():
    log_name = generate_log_name()
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file_path = f'{LOGS_DIR}/{log_name}.log'
    return log_name, log_file_path

def create_new_logger():
    curr_log_file_name, curr_log_file_path = get_curr_log_file_path()
    log_path_dict = {
        'log_name': curr_log_file_name,
        'log_path': curr_log_file_path
    }
    json_write(os.path.join(base_dir, 'logs.json'), log_path_dict)
    logging.basicConfig(filename=curr_log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(curr_log_file_path)

def get_curr_logger():
    log_path_dict = json_read(os.path.join(base_dir, 'logs.json'))
    curr_log_file_path = log_path_dict['log_path']
    logging.basicConfig(filename=curr_log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(curr_log_file_path)

def generate_unique_file_name(file_name):
    now = datetime.now()
    return f'{file_name}_{now.strftime("%Y-%m-%d_%H-%M-%S-%f")}'

def generate_current_time_suffix():
    now = datetime.now()
    return f'{now.strftime("%H-%M-%S-%f")}'

# Google Drive configurations
class GoogleDriveUploader:
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    SERVICE_ACCOUNT_FILE = '/content/drive/My Drive/SD-Data/comfyui-n8n-aici01-7679b55c962b.json'
    DRIVE_FOLDER_ID = '1fZyeDT_eW6ozYXhqi_qLVy-Xnu5JD67a'

    def __init__(self):
        self.drive_service = None
        self._initialize_drive_service()

    def _initialize_drive_service(self):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)
            self.drive_service = build('drive', 'v3', credentials=credentials)
        except Exception as e:
            print(f"Error initializing Drive service: {str(e)}")
            raise RuntimeError(f"Failed to initialize Drive service: {str(e)}")

    def upload_to_drive(self, file_path):
        try:
            file_metadata = {
                'name': os.path.basename(file_path),
                'parents': [self.DRIVE_FOLDER_ID]
            }
            media = MediaFileUpload(file_path, resumable=True)
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            self.drive_service.permissions().create(
                fileId=file.get('id'),
                body={'type': 'anyone', 'role': 'reader'},
                fields='id'
            ).execute()

            file_id = file.get('id')
            return f"https://drive.google.com/uc?id={file_id}"

        except Exception as e:
            raise RuntimeError(f"Failed to upload to Drive: {str(e)}")

# SubtitleNode class
class SubtitleNode:
    def __init__(self):
        self.google_drive_uploader = GoogleDriveUploader()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_file": ("STRING", {"default": ""}),
                "font_name": ("STRING", {"default": "Arial"}),
                "font_size": ("FLOAT", {"default": 24}),
                "font_color": ("STRING", {"default": "FFFFFF"}),  # Màu ở dạng hex, bỏ đi dấu '#'
                "subtitle_position": ("STRING", {"default": "bottom"}),
                "subtitle_style": ("STRING", {"default": "normal"}),
                "translate_to_english": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"

    def process(self, video_file, font_name, font_size, font_color, subtitle_position, subtitle_style, translate_to_english):
        extracted_audio_name = self.extract_audio(video_file)
        params_dict = {
            "translate_to_english": translate_to_english,
            "is_upper": False,
            "eng_font": font_name,
            "font_size": font_size,
            "font_color": font_color,
            "subtitle_position": subtitle_position,
            "subtitle_style": subtitle_style
        }
    
        transcript_text = self.generate_transcript_matrix(extracted_audio_name, params_dict)
        vtt_path = self.convert_transcript_to_subtitles(transcript_text, extracted_audio_name, params_dict)
        
        output_video_path = self.embed_subtitles(
            video_file, 
            vtt_path,  # Đây là file phụ đề (subtitles) đã được tạo ra từ hàm convert_transcript_to_subtitles
            params_dict["eng_font"],  # Truyền tên font từ params_dict
            params_dict["font_size"],  # Truyền kích thước font từ params_dict
            params_dict["font_color"]  # Truyền màu font từ params_dict
        )
    
        # Upload video to Google Drive and get the URL
        output_video_url = self.google_drive_uploader.upload_to_drive(output_video_path)
    
        return (output_video_url,)

    def extract_audio(self, video_file_path):
        if video_file_path.startswith("http://") or video_file_path.startswith("https://"):
            response = requests.get(video_file_path)
            if response.status_code == 200:
                temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video_file.write(response.content)
                video_file_path = temp_video_file.name
            else:
                raise RuntimeError(f"Failed to download video from URL: {video_file_path}")

        file_name_with_ext = os.path.basename(video_file_path)
        file_name = generate_unique_file_name(file_name_with_ext.split('.')[0])
        curr_audio_dir = f'{AUDIO_DIR}/{file_name}'
        os.makedirs(curr_audio_dir, exist_ok=True)
        audio_file_name = f'{file_name}.wav'
        audio_file_path = f'{curr_audio_dir}/{audio_file_name}'

        video_clip = VideoFileClip(video_file_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_file_path)
        video_clip.close()

        return file_name

    def generate_transcript_matrix(self, file_name, params_dict):
        curr_audio_dir = f'{AUDIO_DIR}/{file_name}'
        audio_file_name = f'{file_name}.wav'
        audio_file_path = f'{curr_audio_dir}/{audio_file_name}'

        model_name = "large-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_name, device)

        task = 'transcribe'
        if params_dict['translate_to_english']:
            task = 'translate'

        result = model.transcribe(
            audio_file_path,
            task=task,
            word_timestamps=True
        )

        segments = result['segments']
        transcript_matrix = []
        for i in range(len(segments)):
            words = segments[i]["words"]
            current_row = []
            for j in range(len(words)):
                word_instance = {
                    "start_time": int(words[j]["start"] * 1000),
                    "end_time": int(words[j]["end"] * 1000),
                    "word": words[j]["word"]
                }
                current_row.append(word_instance)
            transcript_matrix.append(current_row)

        return transcript_matrix

    def convert_transcript_to_subtitles(self, transcript_matrix, file_name, params_dict):
        lines = ["WEBVTT\n"]
        for i in range(len(transcript_matrix)):
            for j in range(len(transcript_matrix[i])):
                word = transcript_matrix[i][j]["word"]
                start_time = transcript_matrix[i][j]["start_time"]
                end_time = transcript_matrix[i][j]["end_time"]
                lines.append(f"{self.convert_time_for_vtt_and_srt(start_time)} --> {self.convert_time_for_vtt_and_srt(end_time)}\n{word}\n")

        vtt_text = "\n".join(lines)
        curr_subtitles_dir = f'{SUBTITLES_DIR}/{file_name}'
        os.makedirs(curr_subtitles_dir, exist_ok=True)
        vtt_subtitle_path = f'{curr_subtitles_dir}/{file_name}.vtt'

        with open(vtt_subtitle_path, 'w') as f:
            f.write(vtt_text)

        return vtt_subtitle_path

    def embed_subtitles(self, video_file_path, subtitles_file_path, font_name, font_size, font_color):
        temp_output_path = None
    
        try:
            # Tạo file đầu ra tạm thời
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
                temp_output_path = temp_output.name
    
            # Chạy lệnh FFMPEG để nhúng phụ đề vào video
            ffmpeg_cmd = [
                'ffmpeg', '-i', video_file_path,
                '-vf', f"subtitles={subtitles_file_path}:force_style='Fontname={font_name},Fontsize={font_size},PrimaryColour=&H{font_color}&'",
                '-c:a', 'copy',
                '-c:v', 'libx264',
                '-y',  # Ghi đè file nếu đã tồn tại
                temp_output_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
    
            # Kiểm tra xem file đã được tạo thành công hay chưa
            if os.path.exists(temp_output_path):
                return temp_output_path  # Trả về đường dẫn đến file video đầu ra
            else:
                raise FileNotFoundError(f"Output video not found at {temp_output_path}")
    
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFMPEG failed with error: {str(e)}")
    
        except Exception as e:
            raise RuntimeError(f"Failed to embed subtitles: {str(e)}")
    
        finally:
            # Xóa file tạm nếu tồn tại
            if temp_output_path and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)

    def convert_time_for_vtt_and_srt(self, ms):
        seconds = ms // 1000
        milliseconds = ms % 1000
        minutes = seconds // 60
        hours = minutes // 60
        return f"{hours:02}:{minutes%60:02}:{seconds%60:02}.{milliseconds:03}"

# Cập nhật mappings cho node
NODE_CLASS_MAPPINGS = {
    "SubtitleNode": SubtitleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleNode": "Subtitle Node"
}
