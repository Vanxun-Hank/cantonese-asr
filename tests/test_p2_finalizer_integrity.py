import json
import pytest
from cantonese_asr.io import sha256_file
from scripts.finalize_p2_full import REQUIRED, verified_surface


def fixture_surface(root):
    files = []
    for name in REQUIRED[:-1]:
        path = root/name
        path.write_text('{}\n')
        files.append({'path': name, 'sha256': sha256_file(path)})
    row = dict(complete=True, integrity_pass=True, manifest_sha256='frozen',
               decoder='D0_CURRENT', surface='public', step=4371, files=files)
    (root/'evaluation_receipt.json').write_text(json.dumps(row))
    return {'sha256': 'frozen', 'rows': 1}


def test_accepts_only_complete_hash_verified_surface(tmp_path):
    spec = fixture_surface(tmp_path)
    assert verified_surface(tmp_path, spec, 'public', 4371)['complete']
    (tmp_path/'metrics.json').write_text('{"cer":0}')
    with pytest.raises(ValueError, match='unverified artifact'):
        verified_surface(tmp_path, spec, 'public', 4371)


def test_wrong_split_or_row_count_cannot_enter_matrix(tmp_path):
    spec = fixture_surface(tmp_path)
    with pytest.raises(ValueError, match='invalid receipt'):
        verified_surface(tmp_path, spec, 'validation', 4371)
    with pytest.raises(ValueError, match='incorrect row count'):
        verified_surface(tmp_path, {**spec, 'rows': 2}, 'public', 4371)
