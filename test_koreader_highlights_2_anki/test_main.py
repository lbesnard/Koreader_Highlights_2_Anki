import os
import unittest

from koreader_highlights_2_anki.__main__ import (
    parse_lua_highlights_annotations,
    parse_lua_highlights_bookmarks,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class TestParseLuaHighlightsAnnotations(unittest.TestCase):
    def test_parse_lua_highlights_annotations(self):
        # Define the path to the Lua file
        filepath = os.path.join(
            "Ali Abdaal - Feel-Good Productivity_ How to Do More of What Matters to You.sdr/metadata.epub.lua"
        )
        filepath = os.path.join(ROOT_DIR, "resources", filepath)

        # Call the function with the actual Lua file
        result = parse_lua_highlights_annotations(filepath)

        # Expected output structure (modify according to your actual expected output)
        expected_metadata = {
            "title": "Feel-good Productivity : How to Do More of What Matters to You (9781250865052)",
            "authors": "Abdaal, Ali",
            "language": "en-US",
            "entries": [
                {
                    "chapter": "Introduction",
                    "datetime": "2024-08-30 12:31:05",
                    "notes": "when we’re in a positive mood, we tend to consider a broader range "
                    "of actions, be more open to new experiences, and better integrate "
                    "the information we receive",
                    "page": 14,
                },
                {
                    "chapter": "Introduction",
                    "datetime": "2024-08-30 12:31:27",
                    "notes": "feeling good boosts our creativity –",
                    "page": 15,
                },
            ],
        }

        self.assertIsNotNone(result)
        self.assertEqual(result["title"], expected_metadata["title"])
        self.assertEqual(result["authors"], expected_metadata["authors"])
        self.assertEqual(result["language"], expected_metadata["language"])
        self.assertEqual(result["entries"], expected_metadata["entries"])

    def test_parse_lua_highlights_bookmarks(self):
        # Define the path to the Lua file
        filepath = os.path.join(
            "Wallace J. Nichols - Blue Mind_ How Water Makes You Happier_ More Connected and Better at What You Do.sdr/metadata.epub.lua"
        )
        filepath = os.path.join(ROOT_DIR, "resources", filepath)

        # Call the function with the actual Lua file
        result = parse_lua_highlights_bookmarks(filepath)

        # Expected output structure (modify according to your actual expected output)
        expected_metadata = {
            "title": "Blue Mind: How Water Makes You Happier, More Connected and Better at What You Do",
            "authors": "Céline Cousteau\nWallace J. Nichols",
            "language": "en",
            "entries": [
                {
                    "chapter": "3. The Water Premium",
                    "datetime": "2021-02-15 14:28:25",
                    "notes": "The factors that help boost Ryan Howell’s happiness for the longer "
                    "term are the pursuit and attainment of personal goals and the "
                    "adoption of meaningful activities. Suppose that Howell was "
                    "planning",
                    "page": "12",
                }
            ],
        }

        self.assertIsNotNone(result)
        self.assertEqual(result["title"], expected_metadata["title"])
        self.assertEqual(result["authors"], expected_metadata["authors"])
        self.assertEqual(result["language"], expected_metadata["language"])
        self.assertEqual(result["entries"], expected_metadata["entries"])


if __name__ == "__main__":
    unittest.main()
