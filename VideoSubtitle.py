import json
import os
import requests
import torch
import whisper
from moviepy.editor import VideoFileClip
from _utils import AUDIO_DIR, create_new_logger, generate_unique_file_name, get_curr_logger, AUDIO_DIR, Timestamped_word, JSON_DIR, json_write, write_text_file


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


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ExtractAudioFromVideo": ExtractAudioFromVideo,
    "GenerateTranscriptMatrix": GenerateTranscriptMatrix
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractAudioFromVideo": "Extract Audio from Video URL",
    "GenerateTranscriptMatrix": "Generate Transcript Matrix"
}
