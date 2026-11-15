"""Independent audit of MNIST records; deliberately does not import train.py.

Recomputes metrics, summary statistics, checkpoint selection, split indices,
all corruption hashes and a deterministic sample of forward predictions.
It never retrains or overwrites the core experimental records.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def close(a, b):
    assert abs(float(a) - float(b)) < 1e-12, (a, b)


def condition_views(images, labels, condition, cfg):
    """Independent roll-and-zero shift implementation and noise regeneration."""
    if condition.startswith('shift') or condition.startswith('joint'):
        pixels = int(condition[5:]) if condition.startswith('shift') else 2
        moves = [(-pixels, 0), (pixels, 0), (0, -pixels), (0, pixels)]
    else:
        moves = [(0, 0)]
    noisy = condition.startswith('noise') or condition.startswith('joint')
    if noisy:
        std = float(condition.split('_')[1])
        rng = np.random.default_rng(cfg['test']['joint_noise_seed'] if condition.startswith('joint') else cfg['test']['noise_seed'])
    for dy, dx in moves:
        for start in range(0, len(images), cfg['evaluation_batch_size']):
            view = images[start:start + cfg['evaluation_batch_size']].astype(np.float32) / np.float32(255)
            if dy or dx:
                view = np.roll(view, (dy, dx), axis=(1, 2))
                if dy < 0:
                    view[:, dy:, :] = 0
                elif dy > 0:
                    view[:, :dy, :] = 0
                if dx < 0:
                    view[:, :, dx:] = 0
                elif dx > 0:
                    view[:, :, :dx] = 0
            flat = view.reshape(len(view), 784)
            if noisy:
                flat = np.clip(flat + std * rng.normal(size=flat.shape).astype(np.float32), 0, 1)
            yield flat, labels[start:start + len(flat)]


def main():
    started = time.perf_counter()
    cfg = read('protocol.json')
    env = read('results/environment.json')
    records = read('results/metrics.json')
    summaries = read('results/summary.json')
    methods, seeds, conditions = cfg['methods'], cfg['seeds'], cfg['test']['conditions']
    assert len(records) == 140 and len(summaries) == 28 and len(env['runs']) == 20
    indexed = {(r['method'], r['seed'], r['condition']): r for r in records}
    assert len(indexed) == 140
    expected_keys = {(m, s, c) for m in methods for s in seeds for c in conditions}
    assert set(indexed) == expected_keys
    manifest_count = 0
    for line in (ROOT / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        assert digest(ROOT / relative) == expected, relative
        manifest_count += 1
    assert digest(ROOT / 'train.py') == env['training_script_sha256_at_run']
    assert digest(ROOT / 'protocol.json') == env['protocol_sha256']
    assert digest(ROOT / 'data/mnist.npz') == env['dataset_sha256'] == '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'
    with np.load(ROOT / 'data/mnist.npz', allow_pickle=False) as data:
        train_labels = data['y_train'].astype(np.int64)
        test_images = data['x_test']
        test_labels = data['y_test'].astype(np.int64)
    rng = np.random.default_rng(cfg['split']['seed'])
    tr, va = [], []
    for digit in range(10):
        ids = np.where(train_labels == digit)[0]
        ids = ids[rng.permutation(len(ids))]
        va.extend(ids[:500]); tr.extend(ids[500:])
    tr = np.sort(tr); va = np.sort(va)
    with np.load(ROOT / 'data/split_indices.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['train_idx'], tr)
        assert np.array_equal(saved['validation_idx'], va)
    assert len(tr) == 55000 and len(va) == 5000
    assert not np.intersect1d(tr, va).size
    assert np.array_equal(np.sort(np.r_[tr, va]), np.arange(60000))
    assert np.array_equal(np.bincount(train_labels[va]), [500] * 10)
    assert np.array_equal(np.bincount(train_labels[tr]), env['training_class_counts'])
    assert np.array_equal(np.bincount(test_labels), env['official_test_class_counts'])
    run_index = {(r['method'], r['seed']): r for r in env['runs']}
    metric_count = 0
    selected_epochs = {}
    for method in methods:
        for seed in seeds:
            stem = f'{method}_seed{seed}'
            with open(ROOT / f'results/logs/{stem}.csv', newline='', encoding='utf-8') as f:
                logs = list(csv.DictReader(f))
            assert len(logs) == 10
            assert [int(r['epoch']) for r in logs] == list(range(1, 11))
            for row in logs:
                score = statistics.mean(float(row['val_' + c]) for c in cfg['validation']['conditions'])
                close(row['validation_score'], score)
                assert np.isfinite(float(row['train_loss']))
            best = max(logs, key=lambda row: float(row['validation_score']))
            selected_epochs[stem] = int(best['epoch'])
            with np.load(ROOT / f'weights/{stem}.npz', allow_pickle=False) as weights:
                assert int(weights['epoch']) == int(best['epoch'])
                close(weights['validation_score'], best['validation_score'])
                assert str(weights['method']) == method and int(weights['seed']) == seed
                assert weights['W1'].shape == (784, 128) and weights['W2'].shape == (128, 10)
                assert weights['b1'].shape == (128,) and weights['b2'].shape == (10,)
                assert sum(weights[k].size for k in ('W1', 'W2', 'b1', 'b2')) == env['parameter_count'] == 101770
                assert all(np.isfinite(weights[k]).all() for k in ('W1', 'W2', 'b1', 'b2'))
            run = run_index[method, seed]
            assert run['best_epoch'] == int(best['epoch']) and run['steps'] == 4300
            close(run['best_validation_score'], best['validation_score'])
            with np.load(ROOT / f'results/predictions/{stem}.npz', allow_pickle=False) as predictions, np.load(ROOT / f'results/confusions/{stem}.npz', allow_pickle=False) as confusions:
                assert set(predictions.files) == set(conditions) == set(confusions.files)
                for condition in conditions:
                    pred = predictions[condition]
                    labels = np.tile(test_labels, 4 if condition.startswith(('shift', 'joint')) else 1)
                    assert pred.shape == labels.shape and np.all((pred >= 0) & (pred < 10))
                    cm = np.zeros((10, 10), dtype=np.int64)
                    np.add.at(cm, (labels, pred), 1)
                    assert np.array_equal(cm, confusions[condition])
                    correct = int(np.count_nonzero(pred == labels))
                    f1 = []
                    for k in range(10):
                        tp = int(cm[k, k]); fp = int(cm[:, k].sum()) - tp; fn = int(cm[k, :].sum()) - tp
                        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
                    row = indexed[method, seed, condition]
                    assert row['correct'] == correct and row['n_views'] == len(labels)
                    close(row['accuracy'], correct / len(labels))
                    close(row['macro_f1'], statistics.mean(f1))
                    metric_count += 1
    summary_index = {(r['method'], r['condition']): r for r in summaries}
    assert set(summary_index) == {(m, c) for m in methods for c in conditions}
    for (method, condition), row in summary_index.items():
        assert row['n_seeds'] == 5
        for key in ('accuracy', 'macro_f1'):
            values = [indexed[method, seed, condition][key] for seed in seeds]
            close(row[key + '_mean'], statistics.mean(values))
            close(row[key + '_std'], statistics.stdev(values))
    # CSV and JSON are separate deliverables, so verify both representations.
    for csv_name, json_rows in [('metrics', records), ('summary', summaries)]:
        with open(ROOT / f'results/{csv_name}.csv', newline='', encoding='utf-8') as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == len(json_rows)
        for a, b in zip(csv_rows, json_rows):
            assert set(a) == set(b)
            for key in b:
                if isinstance(b[key], str):
                    assert a[key] == b[key]
                else:
                    close(a[key], b[key])
    hashes = read('results/test_input_hashes.json')
    forward_count = 0
    for condition in conditions:
        size = 40000 if condition.startswith(('shift', 'joint')) else 10000
        sample_ids = np.linspace(0, size - 1, 64, dtype=np.int64)
        sample_inputs = []
        image_hash, label_hash = hashlib.sha256(), hashlib.sha256()
        offset = 0
        for x, y in condition_views(test_images, test_labels, condition, cfg):
            image_hash.update(x.astype('<f4', copy=False).tobytes())
            label_hash.update(y.astype('<i8', copy=False).tobytes())
            here = sample_ids[(sample_ids >= offset) & (sample_ids < offset + len(x))] - offset
            if len(here):
                sample_inputs.append(x[here])
            offset += len(x)
        assert hashes[condition] == {'images_sha256': image_hash.hexdigest(), 'labels_sha256': label_hash.hexdigest(), 'n_views': offset}
        sx = np.concatenate(sample_inputs)
        assert len(sx) == 64
        for method in methods:
            for seed in seeds:
                stem = f'{method}_seed{seed}'
                with np.load(ROOT / f'weights/{stem}.npz', allow_pickle=False) as w:
                    hidden = np.maximum(0, np.matmul(sx, w['W1']) + w['b1'])
                    pred = np.argmax(np.matmul(hidden, w['W2']) + w['b2'], axis=1)
                with np.load(ROOT / f'results/predictions/{stem}.npz', allow_pickle=False) as saved:
                    assert np.array_equal(pred, saved[condition][sample_ids]), (stem, condition)
                forward_count += len(pred)
    report = {
        'status': 'PASS', 'audit_imports_training_code': False,
        'numpy': np.__version__, 'manifest_files_verified': manifest_count,
        'official_archive_hash_verified': True, 'training_and_protocol_hashes_match_run': True,
        'checkpoints_and_earliest_best_epoch_verified': 20,
        'metric_rows_recomputed_from_saved_predictions': metric_count,
        'summary_rows_recomputed': len(summaries), 'csv_json_agreement': True,
        'split_indices_independently_regenerated': True, 'split_disjoint_and_complete': True,
        'test_conditions_independently_regenerated_with_exact_hashes': len(conditions),
        'independent_forward_prediction_samples': forward_count,
        'forward_sampling': '64 evenly spaced deterministic view indices per condition per checkpoint',
        'selected_epochs': selected_epochs,
        'wall_seconds': time.perf_counter() - started,
        'limits': 'Same NumPy version and machine; independent implementation checks, not an external-environment replication. Full checkpoint prediction replay is separately covered by verify.py.'
    }
    (ROOT / 'independent_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
