from .SubtitleNode import SubtitleNode, Timestamped_word
NODE_CLASS_MAPPINGS = {
    "SubtitleNode": SubtitleNode,
    "Timestamped_word": Timestamped_word
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleNode": "Subtitle Node",
    "Timestamped_word": "Timestamped word"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
