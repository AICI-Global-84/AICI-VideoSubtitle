import json
import os
import subprocess
import time
import requests
import torch
import whisper
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from moviepy.editor import VideoFileClip
from ._utils import AUDIO_DIR, create_new_logger, generate_unique_file_name, get_curr_logger, Timestamped_word, JSON_DIR, json_write, write_text_file, word_options_index_map, json_read, word_options_json_path, SUBTITLES_DIR, TMP_OUTPUT_DIR, generate_current_time_suffix, video_quality_map, FONTS_JSON_PATH, FONTS_DIR
 

class ExtractAudioFromVideo:
    def __init__(self):
        self.logger = create_new_logger()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"tooltip": "URL của video để trích xuất âm thanh."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_file_name",)
    FUNCTION = "extract_audio"
    CATEGORY = "Audio Processing"

    def download_video(self, url):
        """Tải video từ URL."""
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            video_file_path = os.path.join(AUDIO_DIR, os.path.basename(url))
            with open(video_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return video_file_path
        else:
            self.logger.error(f"Failed to download video: {url} (status code: {response.status_code})")
            return None

    def extract_audio(self, video_url):
        """Trích xuất âm thanh từ video và lưu vào thư mục chỉ định."""
        video_file_path = self.download_video(video_url)
        if not video_file_path:
            return ("",)  # Trả về tuple rỗng nếu không tải video thành công

        file_name_with_ext = os.path.basename(video_file_path)
        file_name = generate_unique_file_name(file_name_with_ext.split('.')[0])

        curr_audio_dir = f'{AUDIO_DIR}/{file_name}'
        os.makedirs(curr_audio_dir, exist_ok=True)
        audio_file_name = f'{file_name}.wav'
        audio_file_path = f'{curr_audio_dir}/{audio_file_name}'

        try:
            video_clip = VideoFileClip(video_file_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_file_path)
            video_clip.close()
            self.logger.info(f"Audio extracted successfully for {video_file_path}")
            print(f"Audio extracted successfully for {video_file_path}")
        except Exception as e:
            self.logger.exception(f"An error occurred: {e}")
            print("An error occurred:", e)

        return (audio_file_name,)


class GenerateTranscriptMatrix:
    def __init__(self):
        self.logger = get_curr_logger()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file_name": ("STRING", {"tooltip": "Tên file âm thanh đã trích xuất."}),
                "translate_to_english": ("BOOLEAN", {"default": False, "tooltip": "Chọn để dịch sang tiếng Anh."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcript_text_file_name",)
    FUNCTION = "generate_transcript"
    CATEGORY = "Audio Processing"

    def generate_transcript(self, audio_file_name, translate_to_english=False):
        self.logger.info(f'Processing audio file: {audio_file_name}')
        print(f'Processing audio file: {audio_file_name}')

        curr_audio_dir = f'{AUDIO_DIR}/{audio_file_name}'
        audio_file_path = f'{curr_audio_dir}/{audio_file_name}.wav'

        model_name = "large-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_name, device)

        task = 'transcribe' if not translate_to_english else 'translate'
        self.logger.info(f'Task: {task} {"to English" if translate_to_english else "to source language"}')

        result = model.transcribe(audio_file_path, task=task, word_timestamps=True)

        segments = result['segments']
        transcript_matrix = []

        for i in range(len(segments)):
            words = segments[i]["words"]
            current_row = []
            for j in range(len(words)):
                word_instance = Timestamped_word(
                    start_time=int(words[j]["start"] * 1000),
                    end_time=int(words[j]["end"] * 1000),
                    word=words[j]["word"][1:],  # Remove leading space
                )
                current_row.append(word_instance)
            transcript_matrix.append(current_row)

        transcript_matrix_2d_list = [
            [word_instance.to_dict() for word_instance in row]
            for row in transcript_matrix
        ]

        curr_json_dir = f'{JSON_DIR}/{audio_file_name}'
        os.makedirs(curr_json_dir, exist_ok=True)
        transcript_matrix_json_name = f'{audio_file_name}_transcript.json'
        transcript_matrix_json_path = f'{curr_json_dir}/{transcript_matrix_json_name}'
        json_write(transcript_matrix_json_path, transcript_matrix_2d_list)

        lines = []
        for i in range(len(transcript_matrix)):
            line = " | ".join(word_instance.word for word_instance in transcript_matrix[i])
            lines.append(line)
        transcript_text = "\n".join(lines)

        transcript_text_file_name = f'{audio_file_name}_tt.txt'
        transcript_text_file_path = f'{curr_json_dir}/{transcript_text_file_name}'
        write_text_file(transcript_text_file_path, transcript_text)

        return (transcript_text_file_name,)


class FormatSubtitles:
    def __init__(self):
        self.logger = get_curr_logger()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_name": ("STRING", {"tooltip": "Tên file phụ đề sẽ được tạo."}),
                "is_upper": ("BOOLEAN", {"default": False, "tooltip": "Chọn để biến tất cả chữ cái thành chữ hoa."}),
                "word_options_key": ("STRING", {"tooltip": "Khóa cho các tùy chọn từ."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("vtt_subtitle_path", "srt_subtitle_path")
    FUNCTION = "format_subtitles"
    CATEGORY = "Subtitles Processing"

    def format_subtitles(self, file_name, is_upper=False, word_options_key="default"):
        self.logger.info(f'Formatting subtitles for file: {file_name}')
        curr_json_dir = f'{JSON_DIR}/{file_name}'
        transcript_json_name = f'{file_name}_transcript.json'
        transcript_json_path = f'{curr_json_dir}/{transcript_json_name}'

        # Convert JSON transcript to transcript matrix
        transcript_matrix = self.transcript_json_to_transcript_matrix(transcript_json_path)

        # Prepare subtitles based on the transcript matrix
        vtt_text, srt_text = self.convert_transcript_to_subtitles(transcript_matrix, file_name, is_upper, word_options_key)

        # Save subtitles to files
        curr_subtitles_dir = f'{SUBTITLES_DIR}/{file_name}'
        os.makedirs(curr_subtitles_dir, exist_ok=True)

        vtt_subtitle_path = f'{curr_subtitles_dir}/{file_name}.vtt'
        srt_subtitle_path = f'{curr_subtitles_dir}/{file_name}.srt'
        write_text_file(vtt_subtitle_path, vtt_text)
        write_text_file(srt_subtitle_path, srt_text)

        self.logger.info(f'Generated subtitles at: {vtt_subtitle_path} and {srt_subtitle_path}')
        return vtt_subtitle_path, srt_subtitle_path

    def dict_to_timestamped_word(self, d):
        return Timestamped_word(
            start_time=d["start_time"],
            end_time=d["end_time"],
            word=d["word"]
        )

    def transcript_json_to_transcript_matrix(self, transcript_json_path):
        with open(transcript_json_path, 'r') as f:
            transcript_matrix_dict = json.load(f)

        return [
            [self.dict_to_timestamped_word(word_dict) for word_dict in row]
            for row in transcript_matrix_dict
        ]

    def convert_time_for_vtt_and_srt(self, time_in_ms, format):
        hours = int(time_in_ms // 3600000)
        minutes = int((time_in_ms % 3600000) // 60000)
        seconds = int((time_in_ms % 60000) // 1000)
        ms = int(time_in_ms % 1000)
        if format == ".vtt":
            time_string = f"{minutes:02}:{seconds:02}.{ms:03}"
        else:
            time_string = f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"
        return time_string

    def convert_transcript_to_subtitles(self, transcript_matrix, file_name, is_upper, word_options_key):
        # ... (Sử dụng logic từ mã của bạn để tạo ra phụ đề từ transcript_matrix)
        lines = []
        vtt_lines = ["WEBVTT\n"]
        srt_lines = []
        srt_index = 1

        # Logic để tạo vtt và srt từ transcript_matrix (giống với phần mã đã cung cấp trước đó)
        # Thêm vào đây mã để xử lý chuyển đổi và thêm các dòng phụ đề vào vtt_lines và srt_lines

        # Trả về nội dung phụ đề vtt và srt
        return "\n".join(vtt_lines), "\n".join(srt_lines)


class EmbedSubtitles:
    def __init__(self):
        self.logger = get_curr_logger()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_video_path": ("STRING", {"tooltip": "Đường dẫn video đầu vào."}),
                "file_name": ("STRING", {"tooltip": "Tên file phụ đề đã được tạo."}),
                "video_quality_key": ("STRING", {"tooltip": "Khóa chất lượng video."}),
                "eng_font": ("STRING", {"tooltip": "Tên font chữ tiếng Anh."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "embed_subtitles"
    CATEGORY = "Subtitles Processing"

    def embed_subtitles(self, input_video_path, file_name, video_quality_key, eng_font):
        start_time = time.time()
        
        self.logger.info(f'Embedding subtitles into video: {input_video_path}')
        
        curr_subtitles_dir = f"{SUBTITLES_DIR}/{file_name}"
        subtitles_path = f"{curr_subtitles_dir}/{file_name}.vtt"
        curr_tmp_output_dir = f"{TMP_OUTPUT_DIR}/{file_name}"
        os.makedirs(curr_tmp_output_dir, exist_ok=True)
        video_ext = "mp4"
        output_video_path = f"{curr_tmp_output_dir}/{file_name[:-16]}_{generate_current_time_suffix()}.{video_ext}"

        crf = video_quality_map[video_quality_key]
        fonts_dict = json_read(FONTS_JSON_PATH)

        font_lang = "english_fonts"
        font_file_name = fonts_dict[font_lang][eng_font]
        font_path = f'{FONTS_DIR}/{font_lang}/{font_file_name}'

        self.logger.info(f'Using font: {eng_font} from path: {font_path}')
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_video_path,  # Input video file
            "-vf", f"subtitles={subtitles_path}:fontsdir={font_path}:force_style='Fontname={eng_font}'",
            '-c:a', 'copy',           # Copy audio codec
            '-c:v', 'libx264',        # Re-encode video codec
            '-preset', 'ultrafast',   # Preset for faster encoding
            '-crf', f'{crf}',         # Constant Rate Factor for quality
            '-y',                      # Overwrite output files without asking
            output_video_path         # Output video file
        ]

        # Run ffmpeg command
        subprocess.run(ffmpeg_cmd)

        end_time = time.time()
        elapsed_time = int(end_time - start_time)
        self.logger.info(f'Time taken to complete embedding: {elapsed_time} seconds')

        self.logger.info('Subtitles were successfully embedded into the input video')
        return output_video_path


class EmbedSubtitles:
    def __init__(self):
        self.logger = get_curr_logger()
        self.drive_service = self.authenticate_google_drive()

    def authenticate_google_drive(self):
        """Authenticate and create a Google Drive API service."""
        SCOPES = ['https://www.googleapis.com/auth/drive']
        credentials_path = '/content/drive/My Drive/SD-Data/comfyui-n8n-aici01-7679b55c962b.json'  # Thay đổi đường dẫn này cho đúng
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES)
        return build('drive', 'v3', credentials=credentials)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_video_path": ("STRING", {"tooltip": "Đường dẫn video đầu vào."}),
                "file_name": ("STRING", {"tooltip": "Tên file phụ đề đã được tạo."}),
                "video_quality_key": ("STRING", {"tooltip": "Khóa chất lượng video."}),
                "eng_font": ("STRING", {"tooltip": "Tên font chữ tiếng Anh."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "embed_subtitles"
    CATEGORY = "Subtitles Processing"

    def upload_to_google_drive(self, video_path):
        """Upload video to Google Drive and return the shared URL."""
        try:
            file_metadata = {'name': os.path.basename(video_path), 'parents': ['1fZyeDT_eW6ozYXhqi_qLVy-Xnu5JD67a']}  # Cập nhật ID thư mục
            media = MediaFileUpload(video_path, mimetype='video/mp4')
            file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

            # Get file ID and create a shareable link
            file_id = file.get('id')
            self.drive_service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()
            return f"https://drive.google.com/uc?id={file_id}"
        except Exception as e:
            self.logger.error(f"An error occurred while uploading to Google Drive: {e}")
            return None

    def embed_subtitles(self, input_video_path, file_name, video_quality_key, eng_font):
        start_time = time.time()
        
        self.logger.info(f'Embedding subtitles into video: {input_video_path}')
        
        curr_subtitles_dir = f"{SUBTITLES_DIR}/{file_name}"
        subtitles_path = f"{curr_subtitles_dir}/{file_name}.vtt"
        curr_tmp_output_dir = f"{TMP_OUTPUT_DIR}/{file_name}"
        os.makedirs(curr_tmp_output_dir, exist_ok=True)
        video_ext = "mp4"
        output_video_path = f"{curr_tmp_output_dir}/{file_name[:-16]}_{generate_current_time_suffix()}.{video_ext}"

        crf = video_quality_map[video_quality_key]
        fonts_dict = json_read(FONTS_JSON_PATH)

        font_lang = "english_fonts"
        font_file_name = fonts_dict[font_lang][eng_font]
        font_path = f'{FONTS_DIR}/{font_lang}/{font_file_name}'

        self.logger.info(f'Using font: {eng_font} from path: {font_path}')
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_video_path,  # Input video file
            "-vf", f"subtitles={subtitles_path}:fontsdir={font_path}:force_style='Fontname={eng_font}'",
            '-c:a', 'copy',           # Copy audio codec
            '-c:v', 'libx264',        # Re-encode video codec
            '-preset', 'ultrafast',   # Preset for faster encoding
            '-crf', f'{crf}',         # Constant Rate Factor for quality
            '-y',                      # Overwrite output files without asking
            output_video_path         # Output video file
        ]

        # Run ffmpeg command
        subprocess.run(ffmpeg_cmd)

        # Upload video to Google Drive
        video_url = self.upload_to_google_drive(output_video_path)
        end_time = time.time()
        elapsed_time = int(end_time - start_time)
        self.logger.info(f'Time taken to complete embedding: {elapsed_time} seconds')

        self.logger.info('Subtitles were successfully embedded into the input video')
        return video_url



# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ExtractAudioFromVideo": ExtractAudioFromVideo,
    "GenerateTranscriptMatrix": GenerateTranscriptMatrix,
    "FormatSubtitles": FormatSubtitles,
    "EmbedSubtitles": EmbedSubtitles
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractAudioFromVideo": "Extract Audio from Video URL",
    "GenerateTranscriptMatrix": "Generate Transcript Matrix",
    "FormatSubtitles": "Format Subtitles",
    "EmbedSubtitles": "Embed Subtitles"
}
