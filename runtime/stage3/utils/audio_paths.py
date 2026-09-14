"""Resolve audio roots at I/O time while preserving manifest and cache identities."""
import json
import os
from pathlib import Path
import torchaudio

def resolve_audio_path(path):
    mapping_file = os.environ.get("MIZAR_AUDIO_PATH_MAP")
    if not mapping_file or not isinstance(path, (str, os.PathLike)):
        return path
    mapping = json.loads(Path(mapping_file).read_text())
    value = str(path)
    for source in sorted(mapping, key=len, reverse=True):
        root = source.rstrip("/")
        if value == root or value.startswith(root + "/"):
            return str(Path(mapping[source]) / value[len(root):].lstrip("/"))
    return path

def load_audio(path, *args, **kwargs):
    return torchaudio.load(resolve_audio_path(path), *args, **kwargs)
