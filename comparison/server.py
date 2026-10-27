import json
import pathlib
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

import cv2

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent
MODELS = ROOT / 'vendor' / 'models'
NAMES = ['face-%02d.png' % i for i in range(1, 31)] + ['negative-%02d.png' % i for i in range(1, 11)]


def sface_results():
    start = time.perf_counter()
    detector = cv2.FaceDetectorYN.create(str(MODELS / 'face_detection_yunet_2023mar.onnx'), '', (640, 640), 0.9, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(MODELS / 'face_recognition_sface_2021dec.onnx'), '')
    enrollment = None
    rows = []
    for name in NAMES:
        t = time.perf_counter()
        path = DATA / name
        row = {'image': name}
        if not path.is_file():
            row['status'] = 'missing'
        else:
            try:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError('Image could not be decoded')
                h, w = image.shape[:2]
                scale = min(1.0, 640.0 / max(h, w))
                image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
                detector.setInputSize((image.shape[1], image.shape[0]))
                _, faces = detector.detect(image)
                if faces is None or len(faces) == 0:
                    row['status'] = 'no face detected'
                elif len(faces) != 1:
                    row['status'] = 'multiple faces detected'
                    row['faces'] = len(faces)
                else:
                    feature = recognizer.feature(recognizer.alignCrop(image, faces[0])).copy()
                    row['detection_score'] = float(faces[0, 14])
                    if name == NAMES[0]:
                        enrollment = feature
                        row['status'] = 'enrolled'
                    elif enrollment is None:
                        row['status'] = 'enrollment unavailable'
                    else:
                        row['score'] = float(recognizer.match(enrollment, feature, cv2.FaceRecognizerSF_FR_COSINE))
                        row['status'] = 'matched' if row['score'] >= 0.363 else 'not matched'
            except Exception as exc:
                row.update(status='processing error', error=str(exc))
        row['ms'] = round((time.perf_counter() - t) * 1000, 1)
        rows.append(row)
    return {'rows': rows, 'elapsed_ms': round((time.perf_counter() - start) * 1000, 1), 'opencv': cv2.__version__}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Host') != '127.0.0.1:18765':
            self.send_error(403)
            return
        path = urlsplit(self.path).path
        if path == '/api/saved-results':
            try:
                files = [sorted(ROOT.glob('results-face-wasm-*.json')),
                         sorted(ROOT.glob('results-sface-*.json'))]
                if not all(files):
                    self.send_error(404, 'Saved results not found')
                    return
                data = {'face': json.loads(files[0][-1].read_text(encoding='utf-8')),
                        'sface': json.loads(files[1][-1].read_text(encoding='utf-8'))}
                body = json.dumps(data, allow_nan=False).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError) as exc:
                self.send_error(500, explain=str(exc))
            return
        if path == '/api/sface':
            try:
                body = json.dumps(sface_results(), allow_nan=False).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self.send_error(500, explain=str(exc))
            return
        if path == '/':
            target = ROOT / 'index.html'
        elif path == '/four-models':
            reports = list((ROOT / 'exploratory-runs').glob('run-*/report.html'))
            if not reports:
                self.send_error(404, 'Four-model report not found')
                return
            target = max(reports, key=lambda report: report.stat().st_mtime_ns)
        elif path == '/multi-report':
            reports = sorted((ROOT / 'multi-person-results').glob('multi-comparison-*.html'),
                             key=lambda report: (report.stem.removesuffix('-visual'), report.stem.endswith('-visual')))
            if not reports:
                self.send_error(404, 'Multi-person report not found')
                return
            target = reports[-1]
        elif re.fullmatch(r'/multi-person/images/person-(?:0[1-9]|1[0-9]|20)-0[1-5]\.png', path):
            target = DATA / path.lstrip('/')
        elif re.fullmatch(r'/images/(?:face-(?:0[1-9]|[12][0-9]|30)|negative-(?:0[1-9]|10))\.png', path):
            target = DATA / path.rsplit('/', 1)[-1]
        elif re.fullmatch(r'/vendor/(?:face-api\.js|models/[a-zA-Z0-9_.-]+)', path):
            target = ROOT / path.lstrip('/')
        elif path == '/app.js':
            target = ROOT / 'app.js'
        else:
            self.send_error(404)
            return
        if not target.is_file():
            self.send_error(404)
            return
        content = target.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json', '.png': 'image/png'}.get(target.suffix, 'application/octet-stream'))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(content)


if __name__ == '__main__':
    print('Open http://127.0.0.1:18765/ (Ctrl+C to stop)', flush=True)
    HTTPServer(('127.0.0.1', 18765), Handler).serve_forever()
