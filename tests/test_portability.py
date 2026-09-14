"""Check relocation boundaries and release bookkeeping without loading weights."""
import importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
class PortabilityTest(unittest.TestCase):
    def test_audio_root_mapping_preserves_path_boundaries(self):
        helper=ROOT/'runtime/stage3/utils/audio_paths.py'
        spec=importlib.util.spec_from_file_location('test_audio_paths',helper)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'torchaudio':types.SimpleNamespace()}):spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            mapping=Path(td)/'paths.json';mapping.write_text(json.dumps({'/old/audio':'/new/audio','/old/audio/special':'/special'}))
            with patch.dict(os.environ,{'MIZAR_AUDIO_PATH_MAP':str(mapping)}):
                self.assertEqual(module.resolve_audio_path('/old/audio/special/a.wav'),'/special/a.wav')
                self.assertEqual(module.resolve_audio_path('/old/audio/a.wav'),'/new/audio/a.wav')
                self.assertEqual(module.resolve_audio_path('/old/audio-other/a.wav'),'/old/audio-other/a.wav')
            with patch.dict(os.environ,{'MIZAR_AUDIO_PATH_MAP':''}):
                self.assertEqual(module.resolve_audio_path('/old/audio/a.wav'),'/old/audio/a.wav')
    def test_release_changes_have_correct_hashes(self):
        import hashlib
        changes=json.loads((ROOT/'provenance/release_patches.json').read_text())
        for r in changes:
            self.assertEqual(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest(),r['release_sha256'])
    def test_adqa_clean_files_match_frozen_ids(self):
        d=ROOT/'data/evaluation';mask=json.loads((d/'ADQA_DEV_CLEAN_V1.json').read_text());ids=set(mask['kept_ids'])
        rows=json.loads((d/'adqa_clean.json').read_text())
        gold=[json.loads(line) for line in (d/'adqa_clean_ground_truth.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1577);self.assertEqual(len(gold),1577)
        self.assertEqual({r['id'] for r in rows},ids);self.assertEqual({r['id'] for r in gold},ids)
if __name__=='__main__':unittest.main()
