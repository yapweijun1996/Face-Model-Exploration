import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('multi_compare', ROOT / 'multi-compare.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def test_manifest_missing_labels_and_path_escape_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = pathlib.Path(directory) / 'manifest.csv'
            manifest.write_text('person_id,image,role,scenario,identity_review\n'
                                'a,missing.png,enrollment,front,pending\n'
                                ',../outside.png,probe,,\n', encoding='utf-8')
            rows, warnings, digest = comparison.load_manifest(manifest)
            self.assertEqual(rows[0]['file_status'], 'missing')
            self.assertEqual(rows[1]['file_status'], 'invalid path')
            self.assertIn('missing person_id', rows[1]['issues'])
            self.assertIn('missing identity_review', rows[1]['issues'])
            self.assertTrue(warnings)
            self.assertEqual(len(digest), 64)

    def test_all_enrollments_and_missing_intended_enrollment(self):
        rows = [dict(line=2, role='enrollment', person_id='a', image='a.png'),
                dict(line=3, role='enrollment', person_id='b', image='b.png'),
                dict(line=4, role='probe', person_id='a', image='a-probe.png'),
                dict(line=5, role='probe', person_id='c', image='missing.png')]
        values = [{'status': 'detected', 'feature': [1., 0.]},
                  {'status': 'detected', 'feature': [0., 1.]},
                  {'status': 'detected', 'feature': [1., 0.]}, {'status': 'missing'}]
        engines = {'face_wasm': {'rows': values}, 'sface': {'rows': values}}
        pairs = comparison.make_comparisons(rows, engines)
        self.assertEqual(len(pairs), 5)
        self.assertEqual([p['relation'] for p in pairs[:2]], ['intended same', 'intended different'])
        self.assertEqual(pairs[0]['engines']['face_wasm']['score'], 0.0)
        self.assertEqual(pairs[0]['engines']['sface']['score'], 1.0)
        self.assertEqual(pairs[1]['engines']['face_wasm']['status'], 'not matched')
        self.assertEqual(pairs[-1]['engines']['sface']['status'], 'unavailable')
        self.assertIn('missing enrollment label', pairs[-1]['engines']['sface']['reason'])
        self.assertTrue(all(p['identity_labels'] == 'unverified' for p in pairs))


if __name__ == '__main__':
    unittest.main()
