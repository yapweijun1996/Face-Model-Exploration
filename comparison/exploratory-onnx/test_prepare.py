import unittest
import numpy as np
import prepare


class PreparationTests(unittest.TestCase):
    def test_reference_and_transformed_landmarks(self):
        transform = prepare.similarity_transform(prepare.REFERENCE)
        np.testing.assert_allclose(transform, [[1, 0, 0], [0, 1, 0]], atol=1e-10)
        moved = prepare.REFERENCE * 2 + [15, -7]
        transform = prepare.similarity_transform(moved)
        actual = np.column_stack([moved, np.ones(5)]) @ transform.T
        np.testing.assert_allclose(actual, prepare.REFERENCE, atol=1e-10)

    def test_degenerate_landmarks(self):
        with self.assertRaises(ValueError):
            prepare.similarity_transform(np.zeros((5, 2)))

    def test_invalid_row_reports_error_without_detection(self):
        rows = [{'issues': [], 'file_status': 'missing'}]
        self.assertEqual(prepare.prepare(rows, None), [{'status': 'error', 'error': 'missing'}])


if __name__ == '__main__':
    unittest.main()
