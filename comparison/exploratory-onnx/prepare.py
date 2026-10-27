"""Read a validated manifest with the existing loader and prepare YuNet crops in memory."""
import base64
import importlib.util
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('multi_compare', ROOT / 'multi-compare.py')
multi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi)

# ArcFace 112-pixel reference, also used by the selected UniFace EdgeFace export.
REFERENCE = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                      [56.0252, 71.7366], [41.5493, 92.3655],
                      [70.7299, 92.2041]], dtype=np.float64)


def similarity_transform(points):
    """Least-squares similarity transform; reject degenerate landmarks."""
    p = np.asarray(points, dtype=np.float64).reshape(5, 2)
    if not np.isfinite(p).all():
        raise ValueError('non-finite landmarks')
    matrix = np.zeros((10, 4), dtype=np.float64)
    matrix[0::2] = np.column_stack((p[:, 0], -p[:, 1], np.ones(5), np.zeros(5)))
    matrix[1::2] = np.column_stack((p[:, 1], p[:, 0], np.zeros(5), np.ones(5)))
    params, _, rank, _ = np.linalg.lstsq(matrix, REFERENCE.reshape(-1), rcond=None)
    a, b, tx, ty = params
    if rank != 4 or a * a + b * b < 1e-12:
        raise ValueError('degenerate landmarks')
    return np.array([[a, -b, tx], [b, a, ty]])


def prepare(rows, detector):
    results = []
    for row in rows:
        try:
            if row['issues'] or row['file_status'] != 'present':
                raise ValueError('; '.join(row['issues']) or row['file_status'])
            image = cv2.imdecode(np.fromfile(row['_path'], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('image decode failed')
            height, width = image.shape[:2]
            scale = min(1.0, 640.0 / max(height, width))
            if scale < 1:
                image = cv2.resize(image, (round(width * scale), round(height * scale)))
            detector.setInputSize((image.shape[1], image.shape[0]))
            _, faces = detector.detect(image)
            count = 0 if faces is None else len(faces)
            if count != 1:
                raise ValueError(f'expected one face; detected {count}')
            face = faces[0]
            transform = similarity_transform(face[4:14])
            crop = cv2.warpAffine(image, transform, (112, 112), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results.append({'status': 'detected', 'detection_score': float(face[14]),
                            'rgb': base64.b64encode(rgb.tobytes()).decode('ascii')})
        except (ValueError, cv2.error, OSError) as exc:
            results.append({'status': 'error', 'error': str(exc)})
    return results


def main():
    args = json.load(sys.stdin)
    rows, warnings, digest = multi.load_manifest(pathlib.Path(args['manifest']).resolve())
    if digest != args['expected_sha256']:
        raise ValueError('manifest SHA-256 does not match prior results')
    detector = cv2.FaceDetectorYN.create(str(ROOT / 'vendor/models/face_detection_yunet_2023mar.onnx'), '',
                                         (320, 320), 0.9, 0.3, 5000)
    results = prepare(rows, detector)
    print(json.dumps({'manifest_sha256': digest, 'warnings': warnings,
                      'python': sys.version.split()[0], 'opencv': cv2.__version__,
                      'numpy': np.__version__, 'rows': results}, allow_nan=False))


if __name__ == '__main__':
    main()
