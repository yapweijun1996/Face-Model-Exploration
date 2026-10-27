import hashlib
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
ZOO = '47534e27c9851bb1128ccc0102f1145e27f23f98'
BASE = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/'
FILES = {
    'vendor/face-api.js': (BASE + 'dist/face-api.js', None),
    **{f'vendor/models/{name}': (BASE + 'model/' + name, None) for name in [
        'ssd_mobilenetv1_model-weights_manifest.json', 'ssd_mobilenetv1_model.bin',
        'face_landmark_68_model-weights_manifest.json', 'face_landmark_68_model.bin',
        'face_recognition_model-weights_manifest.json', 'face_recognition_model.bin']},
    'vendor/models/face_detection_yunet_2023mar.onnx': (f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{ZOO}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
    'vendor/models/face_recognition_sface_2021dec.onnx': (f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{ZOO}/models/face_recognition_sface/face_recognition_sface_2021dec.onnx', '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'),
}
for name, (url, expected) in FILES.items():
    path = ROOT / name
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and digest != expected:
            raise SystemExit(f'Existing file has unexpected SHA-256: {path}. Inspect it before retrying.')
        print(f'Existing {name}: sha256={digest}')
        continue
    print(f'Downloading {name} from {url}', flush=True)
    data = urllib.request.urlopen(url, timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if expected and digest != expected:
        raise SystemExit(f'Unexpected SHA-256 for {name}: {digest}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f'  sha256={digest}')

