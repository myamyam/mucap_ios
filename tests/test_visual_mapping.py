import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mucap'))
from mucap.render import audio_text_viz as visual


class SharedVisualTests(unittest.TestCase):
    def setUp(self):
        notes = [dict(onset_time=0, pitch=110, duration=1),
                 dict(onset_time=1, pitch=880, duration=1)]
        self.table = visual.apply_note_mapping(visual.note_table(notes), visual.CAPTION_MAPPING)

    def test_original_pitch_palette_includes_lightness(self):
        low = visual.timeline_token_style('둥', self.table.iloc[0], 0, 1, self.table, 'piano-clear')
        high = visual.timeline_token_style('딩', self.table.iloc[1], 1, 2, self.table, 'piano-clear')
        self.assertEqual(low['color'], 'hsl(260 82% 28%)')
        self.assertEqual(high['color'], 'hsl(355 82% 65%)')
        self.assertEqual(low['offset_y'], 115)
        self.assertEqual(high['offset_y'], -115)
        self.assertNotEqual(low['font_size'], high['font_size'])

    def test_html_and_mobile_share_character_styles(self):
        style = visual.timeline_token_style('두웅', self.table.iloc[0], 0, 2, self.table, 'piano-clear')
        html = visual._timeline_token_html('두웅', self.table.iloc[0], 0, 2, self.table, 'piano-clear')
        self.assertNotEqual(style['characters'][0]['color'], style['characters'][1]['color'])
        for char in style['characters']:
            self.assertIn(f'color: {char["color"]}', html)
            self.assertIn(f'opacity: {char["opacity"]:.2f}', html)
            self.assertIn(f'translateY({char["offset_y"]:.1f}px)', html)


if __name__ == '__main__':
    unittest.main()
