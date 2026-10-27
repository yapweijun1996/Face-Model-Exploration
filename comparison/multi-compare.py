"""Exploratory local comparison of a manifest with Face-API WASM and SFace."""
import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT.parent / 'multi-person' / 'manifest.csv'
HEADERS = ['person_id', 'image', 'role', 'scenario', 'identity_review']
MODEL_NAMES = ['ssd_mobilenetv1_model.bin', 'ssd_mobilenetv1_model-weights_manifest.json',
               'face_landmark_68_model.bin', 'face_landmark_68_model-weights_manifest.json',
               'face_recognition_model.bin', 'face_recognition_model-weights_manifest.json', 'face_detection_yunet_2023mar.onnx',
               'face_recognition_sface_2021dec.onnx']


def load_manifest(path):
    data_root = path.parent.resolve()
    manifest_bytes = path.read_bytes()
    with io.StringIO(manifest_bytes.decode('utf-8-sig'), newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HEADERS:
            raise ValueError('Manifest headers must be exactly: ' + ','.join(HEADERS))
        rows = []
        for line, raw in enumerate(reader, 2):
            row = {key: (raw.get(key) or '').strip() for key in HEADERS}
            row['line'] = line
            row['issues'] = [f'missing {key}' for key in HEADERS if not row[key]]
            if None in raw:
                row['issues'].append('extra CSV columns')
            if row['role'] not in ('enrollment', 'probe'):
                row['issues'].append('invalid role')
            image = pathlib.PurePosixPath(row['image'])
            candidate = None
            if not row['image'] or '\\' in row['image'] or image.is_absolute() or '..' in image.parts or ':' in row['image']:
                row['issues'].append('invalid relative image path')
            else:
                candidate = (data_root / pathlib.Path(*image.parts)).resolve()
                if not candidate.is_relative_to(data_root):
                    row['issues'].append('image path escapes dataset')
                    candidate = None
            row['_path'] = str(candidate) if candidate else None
            row['file_status'] = 'invalid path' if not candidate else ('present' if candidate.is_file() else 'missing')
            rows.append(row)
    warnings = []
    people = {r['person_id'] for r in rows if r['person_id']}
    if len(people) != 20:
        warnings.append(f'Expected 20 intended identities; manifest contains {len(people)}')
    for person in sorted(people):
        group = [r for r in rows if r['person_id'] == person]
        enrollment = sum(r['role'] == 'enrollment' for r in group)
        probes = sum(r['role'] == 'probe' for r in group)
        if (enrollment, probes) != (1, 4):
            warnings.append(f'{person}: expected 1 enrollment and 4 probes; found {enrollment} and {probes}')
    images = collections.Counter(r['image'] for r in rows if r['image'])
    warnings.extend(f'Duplicate image path: {name}' for name, count in sorted(images.items()) if count > 1)
    return rows, warnings, hashlib.sha256(manifest_bytes).hexdigest()


def face_wasm(rows):
    request = {'images': [{'path': r['_path'], 'error': '; '.join(r['issues'])} for r in rows]}
    try:
        result = subprocess.run(['node', str(ROOT / 'multi-face-wasm.js')], input=json.dumps(request),
                                text=True, capture_output=True, check=True, cwd=ROOT)
        tagged = [line[len('MULTI_RESULT:'):] for line in result.stdout.splitlines() if line.startswith('MULTI_RESULT:')]
        if len(tagged) != 1:
            raise ValueError('Worker did not return exactly one result')
        output = json.loads(tagged[0])
        if output['backend'] != 'wasm' or len(output['rows']) != len(rows):
            raise ValueError('WASM backend or row count verification failed')
        return output
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        return {'backend': 'unavailable', 'error': detail.strip(),
                'rows': [{'status': 'engine unavailable', 'error': detail.strip()} for _ in rows]}


def sface(rows):
    try:
        import cv2
        detector = cv2.FaceDetectorYN.create(str(ROOT / 'vendor/models/face_detection_yunet_2023mar.onnx'), '', (640, 640), 0.9, 0.3, 5000)
        recognizer = cv2.FaceRecognizerSF.create(str(ROOT / 'vendor/models/face_recognition_sface_2021dec.onnx'), '')
    except Exception as exc:
        return {'backend': 'unavailable', 'error': str(exc),
                'rows': [{'status': 'engine unavailable', 'error': str(exc)} for _ in rows]}
    results = []
    for item in rows:
        row = {}
        try:
            if not item['_path']:
                raise ValueError('Invalid image path')
            path = pathlib.Path(item['_path'])
            if not path.is_file():
                row['status'] = 'missing'
            else:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError('Image could not be decoded')
                height, width = image.shape[:2]
                scale = min(1.0, 640.0 / max(height, width))
                image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
                detector.setInputSize((image.shape[1], image.shape[0]))
                _, faces = detector.detect(image)
                if faces is None or len(faces) == 0:
                    row['status'] = 'no face detected'
                elif len(faces) != 1:
                    row.update(status='multiple faces detected', faces=len(faces))
                else:
                    feature = recognizer.feature(recognizer.alignCrop(image, faces[0])).flatten().tolist()
                    if not all(math.isfinite(x) for x in feature):
                        raise ValueError('Non-finite descriptor')
                    row.update(status='detected', detection_score=float(faces[0, 14]), feature=feature)
        except Exception as exc:
            row.update(status='processing error', error=str(exc))
        results.append(row)
    return {'backend': 'OpenCV CPU', 'opencv': cv2.__version__, 'rows': results}


def score(a, b, kind):
    if len(a) != len(b) or not a:
        raise ValueError('Incompatible descriptors')
    if kind == 'face_wasm':
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    denominator = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    if not denominator:
        raise ValueError('Zero-length descriptor')
    return sum(x*y for x, y in zip(a, b)) / denominator


def make_comparisons(rows, engines):
    pairs = []
    enrollments = [(i, r) for i, r in enumerate(rows) if r['role'] == 'enrollment']
    for index, probe in enumerate(rows):
        if probe['role'] != 'probe':
            continue
        candidates = list(enrollments)
        if probe['person_id'] and not any(e['person_id'] == probe['person_id'] for _, e in candidates):
            candidates.append((None, None))
        for enroll_index, enrollment in candidates:
            same = bool(enrollment and probe['person_id'] and enrollment['person_id'] and probe['person_id'] == enrollment['person_id'])
            relation = 'intended same' if same or enrollment is None else ('intended different' if probe['person_id'] and enrollment['person_id'] else 'unknown: missing identity label')
            pair = {'probe_line': probe['line'], 'probe_image': probe['image'], 'probe_person_id': probe['person_id'],
                    'enrollment_line': enrollment['line'] if enrollment else None,
                    'enrollment_image': enrollment['image'] if enrollment else None,
                    'enrollment_person_id': enrollment['person_id'] if enrollment else probe['person_id'],
                    'relation': relation, 'identity_labels': 'unverified', 'engines': {}}
            for name, engine in engines.items():
                probe_result = engine['rows'][index]
                enroll_result = engine['rows'][enroll_index] if enroll_index is not None else None
                if not enroll_result or 'feature' not in enroll_result or 'feature' not in probe_result:
                    pair['engines'][name] = {'status': 'unavailable', 'reason': f"probe: {probe_result['status']}; enrollment: {enroll_result['status'] if enroll_result else 'missing enrollment label'}"}
                    continue
                try:
                    value = score(probe_result['feature'], enroll_result['feature'], name)
                    if not math.isfinite(value):
                        raise ValueError('Non-finite score')
                    matched = value <= 0.6 if name == 'face_wasm' else value >= 0.363
                    pair['engines'][name] = {'status': 'matched' if matched else 'not matched', 'score': value}
                except ValueError as exc:
                    pair['engines'][name] = {'status': 'unavailable', 'reason': str(exc)}
            pairs.append(pair)
    return pairs


def build_result(manifest):
    rows, warnings, manifest_sha256 = load_manifest(manifest)
    engines = {'face_wasm': face_wasm(rows), 'sface': sface(rows)}
    pairs = make_comparisons(rows, engines)
    for engine in engines.values():
        for row in engine['rows']:
            row.pop('feature', None)
    for row in rows:
        row.pop('_path')
    models = {}
    for name in MODEL_NAMES:
        path = ROOT / 'vendor/models' / name
        models[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {'description': 'Exploratory comparison; intended identity labels are unverified; no separate calibration and evaluation sets are assumed.',
            'manifest_sha256': manifest_sha256, 'model_sha256': models,
            'thresholds': {'face_wasm': {'metric': 'Euclidean distance', 'match': '<= 0.6'}, 'sface': {'metric': 'cosine similarity', 'match': '>= 0.363'}},
            'settings': {'face_detector': 'SSD MobileNet V1 confidence 0.5', 'sface_detector': 'YuNet confidence 0.9, max dimension 640'},
            'warnings': warnings, 'manifest': rows, 'engines': engines, 'comparisons': pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output-dir', type=pathlib.Path, default=ROOT / 'multi-person-results')
    args = parser.parse_args()
    result = build_result(args.manifest.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H-%M-%S-%fZ')
    base = args.output_dir / f'multi-comparison-{timestamp}'
    encoded = json.dumps(result, indent=2, allow_nan=False)
    template = (ROOT / 'multi-report.html').read_text(encoding='utf-8')
    report = template.replace('/* RESULT_DATA */ null', encoded.replace('<', '\\u003c'))
    with base.with_suffix('.json').open('x', encoding='utf-8') as stream:
        stream.write(encoded + '\n')
    with base.with_suffix('.html').open('x', encoding='utf-8') as stream:
        stream.write(report)
    print(f'Saved {base}.json and {base}.html')
    incomplete = any(row['issues'] for row in result['manifest']) or any(
        row['status'] != 'detected' for engine in result['engines'].values() for row in engine['rows']) or bool(result['warnings'])
    return 1 if incomplete else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, csv.Error) as exc:
        print(f'Comparison failed: {exc}', file=sys.stderr)
        sys.exit(1)
