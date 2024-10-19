from .SubtitleNode import SubtitleNode, VideoSubtitle
NODE_CLASS_MAPPINGS = {
    "SubtitleNode": SubtitleNode,
    "VideoSubtitle": VideoSubtitle
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleNode": "Subtitle Node",
    "VideoSubtitle": "Video Subtitle Generator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
