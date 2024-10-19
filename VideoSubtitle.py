import os
import json
import requests
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

class VideoSubtitle:
    def __init__(self):
        self.drive_service = self.authenticate_google_drive()
        self.font_dir = "/content/resources/fonts/english_fonts/"  # Thay đổi đường dẫn này nếu cần

    def authenticate_google_drive(self):
        """Xác thực Google Drive API."""
        SCOPES = ['https://www.googleapis.com/auth/drive']
        credentials_path = '/content/drive/My Drive/SD-Data/comfyui-n8n-aici01-7679b55c962b.json'  # Thay đổi đường dẫn này cho đúng
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES)
        return build('drive', 'v3', credentials=credentials)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": "https://example.com/video.mp4", "tooltip": "URL của video."}),
                "subtitle_text": ("STRING", {"default": "Your subtitle text here", "tooltip": "Text phụ đề."}),
                "fontname": (["Pricedown", "Komika Axis", "Bungee", "Kalam"], {"default": "Pricedown", "tooltip": "Chọn font cho phụ đề."}),
                "fontsize": ("INT", {"default": 20, "min": 10, "max": 100, "step": 1, "tooltip": "Kích cỡ font chữ."}),
                "primary_color": ("STRING", {"default": "&H00FFFFFF", "tooltip": "Màu chính của phụ đề."}),
                "secondary_color": ("STRING", {"default": "&H000000FF", "tooltip": "Màu phụ của phụ đề."}),
                "outline_color": ("STRING", {"default": "&H80000000", "tooltip": "Màu viền của phụ đề."}),
                "back_color": ("STRING", {"default": "&H40000000", "tooltip": "Màu nền phụ đề."}),
                "bold": ("INT", {"default": 0, "tooltip": "In đậm (1 là bật, 0 là tắt)."}),
                "italic": ("INT", {"default": 0, "tooltip": "In nghiêng (1 là bật, 0 là tắt)."}),
                "underline": ("INT", {"default": 0, "tooltip": "Gạch chân (1 là bật, 0 là tắt)."}),
                "uppercase": ("INT", {"default": 0, "tooltip": "Viết hoa (1 là bật, 0 là tắt)."}),
                "outline": ("INT", {"default": 2, "min": 0, "max": 10, "tooltip": "Độ dày viền chữ."}),
                "alignment": ("INT", {"default": 2, "tooltip": "Căn chỉnh văn bản (2 là giữa, 7 là dưới)."}),
                "marginL": ("INT", {"default": 20, "min": 0, "max": 100, "tooltip": "Lề trái."}),
                "marginR": ("INT", {"default": 20, "min": 0, "max": 100, "tooltip": "Lề phải."}),
                "marginV": ("INT", {"default": 20, "min": 0, "max": 100, "tooltip": "Lề dọc."}),
                "max_line_count": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Số dòng tối đa."}),
                "max_words_per_line": ("INT", {"default": 10, "min": 1, "max": 20, "tooltip": "Số từ tối đa trên một dòng."}),
                "karaoke": ("INT", {"default": 0, "tooltip": "Chế độ Karaoke (1 là bật, 0 là tắt)."})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_subtitle"
    OUTPUT_NODE = True
    CATEGORY = "video"

    def upload_to_google_drive(self, video_path):
        """Upload video lên Google Drive và trả về URL công khai."""
        try:
            file_metadata = {'name': os.path.basename(video_path), 'parents': ['1fZyeDT_eW6ozYXhqi_qLVy-Xnu5JD67a']}  # ID folder
            media = MediaFileUpload(video_path, mimetype='video/mp4')
            file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

            file_id = file.get('id')
            self.drive_service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()
            return f"https://drive.google.com/uc?id={file_id}"
        except Exception as e:
            print(f"Upload thất bại: {e}")
            return None

    def generate_subtitle(self, video_url, subtitle_text, fontname, fontsize, primary_color, secondary_color, outline_color, back_color, 
                          bold, italic, underline, uppercase, outline, alignment, marginL, marginR, marginV, max_line_count, 
                          max_words_per_line, karaoke):
        """Xử lý và tạo phụ đề cho video từ URL."""
        video_path = "/tmp/input_video.mp4"
        subtitle_video_path = "/tmp/output_video_with_subtitle.mp4"
        
        # Download video from URL
        response = requests.get(video_url)
        with open(video_path, 'wb') as f:
            f.write(response.content)
        
        # Video processing and subtitle rendering
        video = VideoFileClip(video_path)
        font_path = os.path.join(self.font_dir, f"{fontname}.ttf")  # Tạo đường dẫn đầy đủ cho font

        # Uppercase text if required
        if uppercase:
            subtitle_text = subtitle_text.upper()

        # Tạo TextClip cho phụ đề
        subtitle = TextClip(subtitle_text, fontsize=fontsize, font=font_path, color=primary_color).set_position(('center', 'bottom')).set_duration(video.duration)
        
        # Chèn phụ đề vào video
        final_video = CompositeVideoClip([video, subtitle])
        final_video.write_videofile(subtitle_video_path, codec='libx264', audio_codec='aac')

        # Upload video output lên Google Drive
        public_url = self.upload_to_google_drive(subtitle_video_path)

        return (public_url,)

# Cấu hình node và hiển thị
NODE_CLASS_MAPPINGS = {
    "VideoSubtitle": VideoSubtitle
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSubtitle": "Video Subtitle Generator"
}
