import json
import hashlib
import math
import random
import torch
import torchaudio
from utils.audio_paths import load_audio
from torch.utils.data import Dataset
import os
import random
from pathlib import Path
from tqdm import tqdm
from scipy.signal import fftconvolve
import numpy as np
from transformers import AutoTokenizer
from data.template import DETAIL, WORD, BOTH, FIRST, SECOND, LONGLINE, BOTH_SPK, FIRST_SPK, SECOND_SPK
from data.template import DETAILONLY, WORDONLY, LONGLINEONLY
from data.lazy_memmap import LazyMemmap

class AudioTextEvalDataset(Dataset):
    """Can sample data from audio-text databases
    Params:
    sampling_rate: audio sampling rate
    max_clip_len: max length (seconds) of audio clip to be sampled
    """
    def __init__(
        self,
        data_path="",
        datafiles=[''],
        sampling_rate=44100, 
        max_clip_len=10,
        tokenizer_type="gpt2",
        ip_text_len=40,
        op_text_len=300,
        audio_mode="normal",
        audio_length_policy="fixed",
        ced_hidden_cache=None,
        beats_hidden_cache=None,
    ):
        self.collect_data_jsons(data_path, datafiles)
        self.ip_text_len = ip_text_len
        self.op_text_len = op_text_len
        self.audio_mode = audio_mode
        self.audio_length_policy = str(audio_length_policy or "fixed").lower()
        self.variable_audio_length = self.audio_length_policy in {
            "variable",
            "raw",
            "no_crop",
            "no_crop_no_pad",
        }

        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate
        self.tokenizer_type = tokenizer_type
        self.tokenizer = self._create_tokenizer(tokenizer_type)
        self.ced_hidden_cache = self._load_ced_hidden_cache(ced_hidden_cache)
        self.beats_hidden_cache = self._load_beats_hidden_cache(beats_hidden_cache)
        if self.ced_hidden_cache is not None and self.beats_hidden_cache is not None:
            raise ValueError("Use only one hidden cache: ced_hidden_cache or beats_hidden_cache")
    
    def _create_tokenizer(self, tokenizer_type):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        # if 'gpt' in tokenizer_type:
        tokenizer.add_special_tokens({'pad_token': '!'})
        return tokenizer

    def _load_ced_hidden_cache(self, cache_config):
        if not cache_config:
            return None
        if isinstance(cache_config, str):
            config = {}
            roots = [Path(cache_config)]
        else:
            config = dict(cache_config)
            root_value = config.get("root")
            roots_value = config.get("roots")
            if root_value and roots_value:
                raise ValueError(
                    "data.ced_hidden_cache accepts either root or roots, not both"
                )
            if roots_value is not None:
                if (
                    not isinstance(roots_value, (list, tuple))
                    or not roots_value
                    or any(not str(value) for value in roots_value)
                ):
                    raise ValueError(
                        "data.ced_hidden_cache.roots must be a non-empty list"
                    )
                roots = [Path(str(value)) for value in roots_value]
            elif root_value:
                roots = [Path(str(root_value))]
            else:
                raise ValueError(
                    "data.ced_hidden_cache.root or .roots is required"
                )
        strict = bool(config.get("strict", False))
        expected_policy = str(config.get("policy", ""))
        expected_cache_point = str(config.get("cache_point", ""))
        if strict and len(set(map(str, roots))) != len(roots):
            raise ValueError("Duplicate data.ced_hidden_cache.roots entry")
        shard_roots = []
        for root in roots:
            if not root.exists():
                raise FileNotFoundError(f"CED hidden cache root not found: {root}")
            discovered = (
                [root]
                if (root / "metadata.json").exists()
                else sorted(p for p in root.iterdir() if (p / "metadata.json").exists())
            )
            if not discovered:
                raise FileNotFoundError(
                    f"No CED hidden cache metadata found under {root}"
                )
            shard_roots.extend(discovered)
        shards = []
        key_to_index = {}
        reference_policy = None
        reference_cache_point = None
        reference_hidden_shape = None
        for shard_id, shard_root in enumerate(shard_roots):
            metadata = json.loads((shard_root / "metadata.json").read_text())
            shape = tuple(metadata["shape"])
            if strict:
                if len(shape) != 3 or any(int(value) <= 0 for value in shape):
                    raise ValueError(f"Invalid CED cache shape in {shard_root}: {shape}")
                if expected_policy and metadata.get("policy") != expected_policy:
                    raise ValueError(
                        f"CED cache policy mismatch in {shard_root}: "
                        f"{metadata.get('policy')} != {expected_policy}"
                    )
                if expected_cache_point and metadata.get("cache_point") != expected_cache_point:
                    raise ValueError(
                        f"CED cache point mismatch in {shard_root}: "
                        f"{metadata.get('cache_point')} != {expected_cache_point}"
                    )
                policy = metadata.get("policy")
                cache_point = metadata.get("cache_point")
                hidden_shape = shape[1:]
                if reference_policy is None:
                    reference_policy = policy
                    reference_cache_point = cache_point
                    reference_hidden_shape = hidden_shape
                elif (
                    policy != reference_policy
                    or cache_point != reference_cache_point
                    or hidden_shape != reference_hidden_shape
                ):
                    raise ValueError(
                        "Incompatible CED hidden cache roots: policy, cache point, "
                        f"or non-batch shape differs in {shard_root}"
                    )
                if int(metadata.get("num_cached_this_shard", -1)) != int(shape[0]):
                    raise ValueError(f"CED cache row-count mismatch in {shard_root}")
                hidden_path = shard_root / "hidden.f16.dat"
                expected_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
                if not hidden_path.is_file() or hidden_path.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"CED cache hidden byte-size mismatch in {shard_root}"
                    )
            valid_tokens = np.load(shard_root / "valid_tokens.npy")
            segment_lengths = np.load(shard_root / "segment_lengths.npy")
            if strict:
                if valid_tokens.shape != (shape[0],):
                    raise ValueError(f"CED valid_tokens shape mismatch in {shard_root}")
                if segment_lengths.ndim != 2 or segment_lengths.shape[0] != shape[0]:
                    raise ValueError(f"CED segment_lengths shape mismatch in {shard_root}")
                if bool(((valid_tokens <= 0) | (valid_tokens > shape[1])).any()):
                    raise ValueError(f"CED valid_tokens values are invalid in {shard_root}")
            shards.append({
                "root": shard_root,
                "metadata": metadata,
                "hidden": LazyMemmap(
                    shard_root / "hidden.f16.dat",
                    dtype=np.float16,
                    mode="r",
                    shape=shape,
                ),
                "valid_tokens": valid_tokens,
                "segment_lengths": segment_lengths,
            })
            shard_local_indices = set()
            shard_key_count = 0
            with open(shard_root / "keys.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    global_key = str(row["global_key"])
                    local_index = int(row["local_index"])
                    shard_key_count += 1
                    if strict and global_key in key_to_index:
                        raise ValueError(f"Duplicate CED cache key: {global_key}")
                    if strict and not 0 <= local_index < shape[0]:
                        raise ValueError(
                            f"CED cache local_index out of bounds in {shard_root}: {local_index}"
                        )
                    if strict:
                        if (
                            len(global_key) != 40
                            or any(char not in "0123456789abcdef" for char in global_key)
                        ):
                            raise ValueError(
                                f"Invalid CED cache global_key in {shard_root}: {global_key}"
                            )
                        if local_index in shard_local_indices:
                            raise ValueError(
                                f"Duplicate CED cache local_index in {shard_root}: {local_index}"
                            )
                        shard_local_indices.add(local_index)
                    key_to_index[global_key] = (shard_id, local_index)
            if strict and (
                shard_key_count != shape[0]
                or shard_local_indices != set(range(shape[0]))
            ):
                raise ValueError(
                    f"CED cache key/local-index coverage mismatch in {shard_root}: "
                    f"keys={shard_key_count}, shape_rows={shape[0]}"
                )
        return {
            "root": roots[0],
            "roots": roots,
            "metadata": shards[0]["metadata"],
            "shards": shards,
            "key_to_index": key_to_index,
        }

    def _load_beats_hidden_cache(self, cache_config):
        if not cache_config:
            return None
        if isinstance(cache_config, str):
            root = Path(cache_config)
        else:
            root = Path(cache_config.get("root", ""))
        if not root:
            return None
        shard_roots = [root] if (root / "metadata.json").exists() else sorted(
            p for p in root.iterdir() if (p / "metadata.json").exists()
        )
        if not shard_roots:
            raise FileNotFoundError(f"No BEATs hidden cache metadata found under {root}")
        shards = []
        key_to_index = {}
        for shard_id, shard_root in enumerate(shard_roots):
            metadata = json.loads((shard_root / "metadata.json").read_text())
            shape = tuple(metadata["shape"])
            shards.append({
                "root": shard_root,
                "metadata": metadata,
                "hidden": LazyMemmap(
                    shard_root / "hidden.f16.dat",
                    dtype=np.float16,
                    mode="r",
                    shape=shape,
                ),
                "valid_tokens": np.load(shard_root / "valid_tokens.npy"),
                "segment_lengths": np.load(shard_root / "segment_lengths.npy"),
            })
            with open(shard_root / "keys.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key_to_index[row["global_key"]] = (shard_id, int(row["local_index"]))
        return {
            "root": root,
            "metadata": shards[0]["metadata"],
            "shards": shards,
            "key_to_index": key_to_index,
        }

    def _ced_cache_key(self, filepath1, filepath2):
        policy = self.ced_hidden_cache["metadata"]["policy"]
        raw = f"{policy}\n{filepath1}\n{filepath2}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _beats_cache_key(self, filepath1):
        policy = self.beats_hidden_cache["metadata"]["policy"]
        raw = f"{policy}\n{filepath1}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _tokenize_text(self, sentence, max_len):
        # if 'gpt' in self.tokenizer_type:
        sentence = sentence + ' <|endoftext|>'
        return self.tokenizer.encode_plus(text=sentence, add_special_tokens=True, truncation=True,
                                          max_length=max_len, pad_to_max_length=True, return_tensors="pt")

    def collect_data_jsons(self, data_path, datafiles):
        # collect per class data json
        all_data_json = []
        self.data_path = data_path
        for i, d in enumerate(datafiles):
            with open(d, 'r', encoding='utf-8') as fp:
                data_json = json.load(fp)
                # Only print from rank 0 to avoid duplicate messages
                import os
                local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
                if local_rank == 0:
                    print(f"Dataset: {d.split(os.path.sep)[-1]}, \t Examples: {str(len(data_json))}")
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json
        self.exists_filepaths1 = list(set([self.all_data_json[i]["filepath1"] for i in range(len(self.all_data_json)) if self.all_data_json[i]["filepath1"] != '']))

    def __len__(self):
        return len(self.all_data_json)

    def _cut_or_centercrop(self, waveform):
        # waveform: [1, samples]
        if waveform.shape[0] > 1:
            waveform = (waveform[0] + waveform[1]) / 2
            waveform = waveform.unsqueeze(0)

        if waveform.size(1) > self.max_length:
            start_idx = (waveform.size(1) - self.max_length) // 2
            waveform = waveform[:, start_idx:start_idx+self.max_length]
        else:
            temp_wav = torch.zeros(1, self.max_length)
            temp_wav[:, 0:waveform.size(1)] = waveform
            waveform = temp_wav

        assert waveform.size(1) == self.max_length, \
            f"number of audio samples is {waveform.size(1)}"

        return waveform

    def _to_mono(self, waveform):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _load_audio_clip(self, file_path):
        info = torchaudio.info(file_path)
        audio_rate = info.sample_rate
        if self.variable_audio_length:
            return load_audio(file_path, channels_first=True)
        max_source_frames = int(math.ceil(self.max_length * audio_rate / self.sampling_rate))
        if info.num_frames > max_source_frames > 0:
            frame_offset = (info.num_frames - max_source_frames) // 2
            return load_audio(
                file_path,
                channels_first=True,
                frame_offset=frame_offset,
                num_frames=max_source_frames,
            )
        return load_audio(file_path, channels_first=True)

    def _read_audio(self, index):
        file_path1, file_path2 = self.all_data_json[index]["filepath1"], self.all_data_json[index]["filepath2"]
        if self.all_data_json[index]["filepath2"] == '':
            file_path2 = random.choice(self.exists_filepaths1)
        
        file_path1 = os.path.join(self.data_path, file_path1)
        file_path1 = file_path1.replace("/",os.path.sep).replace("\\",os.path.sep)
        file_path2 = os.path.join(self.data_path, file_path2)
        file_path2 = file_path2.replace("/",os.path.sep).replace("\\",os.path.sep)
        try:
            audio_data1, audio_rate1 = self._load_audio_clip(file_path1)
            audio_data2, audio_rate2 = self._load_audio_clip(file_path2)

            return audio_data1, audio_data2, audio_rate1, audio_rate2, file_path1, file_path2
        
        except Exception as e:
            print(f'error: {e} occurs, when loading {file_path1} or {file_path2}')
            random_index = random.randint(0, len(self.all_data_json)-1)
            return self._read_audio(index=random_index)
        
    def _create_answer_input(self, index):
        input = self.all_data_json[index]['input']
        if input == "explain the difference in few words":
            input = "Explain the difference between the two audios in few words."
            answer = self.all_data_json[index]['answer']
        elif input == "explain the difference in a sentence":
            input = "Explain the difference between the two audios in one extended sentence."
            answer = self.all_data_json[index]['answer']
        elif input == "explain the difference in detail":
            input = "Explain the difference between the two audios in detail."
            answer = self.all_data_json[index]['answer']
        elif input == "caption first audio":
            input = "caption the audio"
            answer = self.all_data_json[index]['caption1']
        else:
            input = self.all_data_json[index]["input"]
            answer = self.all_data_json[index]['answer']
            
        return answer.lower(), input.lower()
    
    def _read_text(self, index):
        answer, input = self._create_answer_input(index)

        tok_answer = self._tokenize_text(answer, self.op_text_len)
        tok_input = self._tokenize_text(input, self.ip_text_len)
        return tok_answer, tok_input, answer, input

    def _read_cached_ced_hidden(self, index):
        row = self.all_data_json[index]
        file_path1 = row["filepath1"]
        file_path2 = row["filepath2"]
        key = self._ced_cache_key(file_path1, file_path2)
        cache_ref = self.ced_hidden_cache["key_to_index"].get(key)
        if cache_ref is None:
            raise KeyError(f"CED hidden cache miss for {file_path1} / {file_path2}")
        shard_id, cache_idx = cache_ref
        shard = self.ced_hidden_cache["shards"][shard_id]
        valid = int(shard["valid_tokens"][cache_idx])
        hidden_np = shard["hidden"][cache_idx, :valid, :]
        seg_np = shard["segment_lengths"][cache_idx]
        return (
            torch.from_numpy(np.array(hidden_np, copy=True)),
            torch.from_numpy(np.array(seg_np, copy=True)).long(),
            os.path.join(self.data_path, file_path1).replace("/", os.path.sep).replace("\\", os.path.sep),
            os.path.join(self.data_path, file_path2).replace("/", os.path.sep).replace("\\", os.path.sep),
        )

    def _read_cached_beats_hidden(self, index):
        row = self.all_data_json[index]
        file_path1 = row["filepath1"]
        file_path2 = row["filepath2"]
        key = self._beats_cache_key(file_path1)
        cache_ref = self.beats_hidden_cache["key_to_index"].get(key)
        if cache_ref is None:
            raise KeyError(f"BEATs hidden cache miss for {file_path1}")
        shard_id, cache_idx = cache_ref
        shard = self.beats_hidden_cache["shards"][shard_id]
        valid = int(shard["valid_tokens"][cache_idx])
        hidden_np = shard["hidden"][cache_idx, :valid, :]
        seg_np = shard["segment_lengths"][cache_idx]
        return (
            torch.from_numpy(np.array(hidden_np, copy=True)),
            torch.from_numpy(np.array(seg_np, copy=True)).long(),
            os.path.join(self.data_path, file_path1).replace("/", os.path.sep).replace("\\", os.path.sep),
            os.path.join(self.data_path, file_path2).replace("/", os.path.sep).replace("\\", os.path.sep),
        )

    def __getitem__(self, index):
        cached_hidden = None
        cached_segment_lengths = None
        cached_beats_hidden = None
        cached_beats_segment_lengths = None
        if self.ced_hidden_cache is not None:
            cached_hidden, cached_segment_lengths, file_path1, file_path2 = self._read_cached_ced_hidden(index)
            audio_data1 = torch.zeros(1, 1)
            audio_data2 = torch.zeros(1, 1)
        elif self.beats_hidden_cache is not None:
            cached_beats_hidden, cached_beats_segment_lengths, file_path1, file_path2 = self._read_cached_beats_hidden(index)
            audio_data1 = torch.zeros(1, 1)
            audio_data2 = torch.zeros(1, 1)
        else:
            # create a audio tensor
            audio_data1, audio_data2, audio_rate1, audio_rate2, file_path1, file_path2 = self._read_audio(index)
        tok_answer, tok_input, answer_text, input_text = self._read_text(index)
        
        # resample audio clip
        if cached_hidden is None:
            try:
                if audio_rate1 != self.sampling_rate:
                    audio_data1 = torchaudio.functional.resample(audio_data1, orig_freq=audio_rate1, new_freq=self.sampling_rate)
                if audio_rate2 != self.sampling_rate:
                    audio_data2 = torchaudio.functional.resample(audio_data2, orig_freq=audio_rate2, new_freq=self.sampling_rate)
            except Exception as e:
                print(f'Error resampling: {e} occurs, when loading {file_path1} or {file_path2}')
        
        # audio_data1 = audio_data1.unsqueeze(0)
        # audio_data2 = audio_data2.unsqueeze(0)
        
        if cached_hidden is None and self.variable_audio_length:
            audio_data1 = self._to_mono(audio_data1)
            if os.path.abspath(file_path1) == os.path.abspath(file_path2):
                audio_data2 = audio_data1.new_zeros(1, 0)
            else:
                audio_data2 = self._to_mono(audio_data2)
        elif cached_hidden is None:
            audio_data1 = self._cut_or_centercrop(audio_data1)
            audio_data2 = self._cut_or_centercrop(audio_data2)

        if self.audio_mode in ("zero_audio", "question_only"):
            audio_data1 = torch.zeros_like(audio_data1)
            audio_data2 = torch.zeros_like(audio_data2)
        elif self.audio_mode != "normal":
            raise ValueError(f"Unknown data.audio_mode: {self.audio_mode}")

        data_dict = {
            'index': index,
            'waveform1': audio_data1,
            'waveform2': audio_data2,
            'answer': tok_answer,
            'answer_text': answer_text,
            'input_text': input_text,
            'input': tok_input,
            'file_path1': file_path1,
            'file_path2': file_path2,
        }
        if cached_hidden is not None:
            data_dict["ced_hidden"] = cached_hidden
            data_dict["ced_hidden_segment_lengths"] = cached_segment_lengths
        if cached_beats_hidden is not None:
            data_dict["beats_hidden"] = cached_beats_hidden
            data_dict["beats_hidden_segment_lengths"] = cached_beats_segment_lengths

        return data_dict
    

def collate_fn(list_data_dict):
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {
                'text': 'a sound of dog',
                'waveform': (1, samples),
                'modality': 'audio_text'
            }
            ...
            ]
    Returns:
        data_dict: e.g. 
            'audio_text': {
                'text': ['a sound of dog', ...]
                'waveform': (batch_size, 1, samples)
        }
    """
    at_data_dict = {}
    
    if len(list_data_dict) > 0:
        for key in list_data_dict[0].keys():
            at_data_dict[key] = [at_data_dict[key] for at_data_dict in list_data_dict]
            if key == 'waveform1' or key == "waveform2":
                lengths = torch.tensor([x.shape[-1] for x in at_data_dict[key]], dtype=torch.long)
                if len(set(lengths.tolist())) == 1:
                    at_data_dict[key] = torch.stack(at_data_dict[key]).squeeze(1)
                else:
                    max_len = int(lengths.max().item())
                    padded = []
                    for waveform in at_data_dict[key]:
                        if waveform.shape[-1] < max_len:
                            pad = torch.zeros(
                                waveform.shape[0],
                                max_len - waveform.shape[-1],
                                dtype=waveform.dtype,
                            )
                            waveform = torch.cat((waveform, pad), dim=-1)
                        padded.append(waveform)
                    at_data_dict[key] = torch.stack(padded).squeeze(1)
                    at_data_dict[f"{key}_lengths"] = lengths
            elif key == "ced_hidden":
                lengths = torch.tensor([x.shape[0] for x in at_data_dict[key]], dtype=torch.long)
                max_len = int(lengths.max().item())
                hidden_dim = int(at_data_dict[key][0].shape[-1])
                padded = []
                for hidden in at_data_dict[key]:
                    if hidden.shape[0] < max_len:
                        pad = torch.zeros(max_len - hidden.shape[0], hidden_dim, dtype=hidden.dtype)
                        hidden = torch.cat((hidden, pad), dim=0)
                    padded.append(hidden)
                at_data_dict[key] = torch.stack(padded)
                at_data_dict["ced_hidden_lengths"] = lengths
            elif key == "ced_hidden_segment_lengths":
                max_len = max(x.shape[0] for x in at_data_dict[key])
                padded = []
                for lengths in at_data_dict[key]:
                    if lengths.shape[0] < max_len:
                        lengths = torch.cat(
                            (lengths, torch.zeros(max_len - lengths.shape[0], dtype=lengths.dtype)),
                            dim=0,
                        )
                    padded.append(lengths)
                at_data_dict[key] = torch.stack(padded)
            elif key == "beats_hidden":
                lengths = torch.tensor([x.shape[0] for x in at_data_dict[key]], dtype=torch.long)
                max_len = int(lengths.max().item())
                hidden_dim = int(at_data_dict[key][0].shape[-1])
                padded = []
                for hidden in at_data_dict[key]:
                    if hidden.shape[0] < max_len:
                        pad = torch.zeros(max_len - hidden.shape[0], hidden_dim, dtype=hidden.dtype)
                        hidden = torch.cat((hidden, pad), dim=0)
                    padded.append(hidden)
                at_data_dict[key] = torch.stack(padded)
                at_data_dict["beats_hidden_lengths"] = lengths
            elif key == "beats_hidden_segment_lengths":
                max_len = max(x.shape[0] for x in at_data_dict[key])
                padded = []
                for lengths in at_data_dict[key]:
                    if lengths.shape[0] < max_len:
                        lengths = torch.cat(
                            (lengths, torch.zeros(max_len - lengths.shape[0], dtype=lengths.dtype)),
                            dim=0,
                        )
                    padded.append(lengths)
                at_data_dict[key] = torch.stack(padded)
            elif key == 'answer' or key == "input":
                stack = {k:[] for k in at_data_dict[key][0].keys()}
                for entry in at_data_dict[key]:
                    for k in entry.keys():
                        stack[k].append(entry[k])
                at_data_dict[key] = {k:torch.stack(stack[k]).squeeze(1) for k in stack.keys()}                       
            elif key == 'file_path1' or key == "file_path2" or key == "answer_text" or key == "input_text":
                at_data_dict[key] = [text for text in at_data_dict[key]]
    
    return at_data_dict
