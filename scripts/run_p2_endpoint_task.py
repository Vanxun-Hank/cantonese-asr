#!/usr/bin/env python3
"""Execute one preselected surface; no model selection or automatic retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--asset-root', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    if plan['selection_surface'] != 'validation':
        raise ValueError('selection must be validation-only')
    task = plan['evaluation_tasks'][a.index]
    config = a.code_root / 'configs/rounds/raw_winner_p2_full.json'
    if sha(config) != plan['config_sha256']:
        raise ValueError('config changed')
    cfg = json.loads(config.read_text())
    model = Path(task['model_dir'])
    complete = json.loads((model.parent / 'COMPLETE.json').read_text())
    for key in ('arm', 'seed', 'step'):
        if complete[key] != task[key]:
            raise ValueError(f'checkpoint identity mismatch: {key}')
    if complete['config_sha256'] != plan['config_sha256']:
        raise ValueError('checkpoint config mismatch')
    # Verify all model files, including tokenizer/config, before loading.
    for entry in complete['files']:
        if entry['path'].startswith('model/'):
            if sha(model.parent / entry['path']) != entry['sha256']:
                raise ValueError(f'checkpoint hash mismatch: {entry["path"]}')
    output = Path(task['output_dir'])
    if output.exists():
        raise FileExistsError(f'preserve existing evidence: {output}')
    processor = a.asset_root / cfg['models'][cfg['arms'][task['arm']]['model']]
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    command = [sys.executable, str(a.code_root/'scripts/evaluate_p2_full.py'),
               '--config', str(config), '--asset-root', str(a.asset_root),
               '--model-dir', str(model), '--processor-dir', str(processor),
               '--output-dir', str(output), '--surface', task['surface'],
               '--step', str(task['step']), '--decoder', 'D0_CURRENT']
    print(json.dumps({'plan_sha256': sha(a.plan), 'task': task, 'command': command}), flush=True)
    subprocess.run(command, cwd=a.code_root, env=env, check=True)


if __name__ == '__main__':
    main()
