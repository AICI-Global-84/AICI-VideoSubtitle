from .SubtitleNode import SubtitleNode
from .VideoSubtitle import ExtractAudioFromVideo, GenerateTranscriptMatrix, FormatSubtitles, EmbedSubtitles

NODE_CLASS_MAPPINGS = {
    "SubtitleNode": SubtitleNode,
    "ExtractAudioFromVideo": ExtractAudioFromVideo,
    "GenerateTranscriptMatrix": GenerateTranscriptMatrix,
    "FormatSubtitles": FormatSubtitles,
    "EmbedSubtitles": EmbedSubtitles
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleNode": "Subtitle Node",
    "ExtractAudioFromVideo": "Extract Audio from Video URL",
    "GenerateTranscriptMatrix": "Generate Transcript Matrix",
    "FormatSubtitles": "Format Subtitles",
    "EmbedSubtitles": "Embed Subtitles"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
