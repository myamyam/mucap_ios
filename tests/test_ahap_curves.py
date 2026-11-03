import unittest

import numpy as np

from multimodal_caption.wav_to_ahap import simplify_curve, parameter_curves


class CurveReductionTests(unittest.TestCase):
    def test_linear_curve_reduces_to_endpoints(self):
        times = np.linspace(0, 2, 201)
        values = times / 2
        reduced_times, reduced_values = simplify_curve(times, values, 0.01)
        np.testing.assert_array_equal(reduced_times, [0, 2])
        np.testing.assert_array_equal(reduced_values, [0, 1])

    def test_sharp_peaks_stay_within_error_bound(self):
        times = np.linspace(0, 3, 301)
        values = np.random.default_rng(42).uniform(0, 0.02, len(times))
        values[[40, 41, 150]] = [1, 0.8, 0.7]
        reduced_times, reduced_values = simplify_curve(times, values, 0.01)
        self.assertLessEqual(np.max(np.abs(values - np.interp(
            times, reduced_times, reduced_values))), 0.01 + 1e-12)
        self.assertIn(times[40], reduced_times)
        self.assertEqual(reduced_times[0], times[0])
        self.assertEqual(reduced_times[-1], times[-1])
        curves = parameter_curves('HapticIntensityControl', reduced_times, reduced_values)
        previous = None
        for entry in curves:
            curve = entry['ParameterCurve']
            points = curve['ParameterCurveControlPoints']
            self.assertLessEqual(len(points), 16)
            if previous is not None:
                self.assertAlmostEqual(previous[0], curve['Time'])
                self.assertEqual(previous[1], points[0]['ParameterValue'])
            previous = (curve['Time'] + points[-1]['Time'], points[-1]['ParameterValue'])

    def test_zero_tolerance_keeps_original(self):
        times = np.linspace(0, 1, 101)
        values = np.zeros(101)
        reduced_times, reduced_values = simplify_curve(times, values, 0)
        np.testing.assert_array_equal(reduced_times, times)
        np.testing.assert_array_equal(reduced_values, values)
