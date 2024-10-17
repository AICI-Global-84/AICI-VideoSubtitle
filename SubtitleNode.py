import os
import subprocess
import whisper
import torch
import time
from moviepy.editor import VideoFileClip
from _utils import AUDIO_DIR, create_new_logger, generate_unique_file_name
from _utils import get_curr_logger, AUDIO_DIR, Timestamped_word, JSON_DIR, json_write, write_text_file
from _utils import (get_curr_logger, JSON_DIR, word_options_index_map, json_read, word_options_json_path, SUBTITLES_DIR,
                    write_text_file)
from _utils import get_curr_logger, SUBTITLES_DIR, TMP_OUTPUT_DIR, generate_current_time_suffix, video_quality_map, \
    json_read, FONTS_JSON_PATH, FONTS_DIR


class SubtitleNode:
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
        # Extract audio from video and generate subtitles
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

        # Tạo transcript từ audio
        transcript_text = self.generate_transcript_matrix(extracted_audio_name, params_dict)

        # Chuyển transcript thành file subtitle (.vtt)
        vtt_path = self.convert_transcript_to_subtitles(transcript_text, extracted_audio_name, params_dict)

        # Chèn subtitle vào video
        output_video = self.embed_subtitles(video_file, extracted_audio_name, params_dict)

        # Trả về đường dẫn tới video đã chèn subtitle
        return (output_video,)

    def extract_audio(self, video_file_path):
        file_name_with_ext = os.path.basename(video_file_path)
        file_name = self.generate_unique_file_name(file_name_with_ext.split('.')[0])
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
        # Tạo phụ đề từ transcript_matrix dưới dạng `.vtt`
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

    def embed_subtitles(self, video_file_path, file_name, params_dict):
        curr_subtitles_dir = f"{SUBTITLES_DIR}/{file_name}"
        subtitles_path = f"{curr_subtitles_dir}/{file_name}.vtt"
        
        output_video_path = f"{TMP_OUTPUT_DIR}/{file_name}_output.mp4"
        font_name = params_dict["eng_font"]
        font_size = params_dict["font_size"]
        font_color = params_dict["font_color"]
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_file_path,
            "-vf", f"subtitles={subtitles_path}:force_style='Fontname={font_name},Fontsize={font_size},PrimaryColour=&H{font_color}&'",
            '-c:a', 'copy',
            '-c:v', 'libx264',
            '-y',  # Overwrite output
            output_video_path
        ]
    
        subprocess.run(ffmpeg_cmd)
        return output_video_path

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
