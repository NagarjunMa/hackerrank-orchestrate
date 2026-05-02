"""
test_indexer_utils.py — Unit tests for pure-function utilities in indexer.py.

No external model downloads. No FAISS. Tests only parsing functions.

Run with:
    python -m pytest code/tests/test_indexer_utils.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer import parse_frontmatter, get_company, get_product_area


class TestParseFrontmatter(unittest.TestCase):

    def test_valid_yaml_frontmatter(self):
        text = "---\ntitle: Test Article\nbreadcrumbs:\n  - Screen\n  - Test Settings\n---\nBody content here."
        meta, body = parse_frontmatter(text)
        self.assertEqual(meta.get("title"), "Test Article")
        self.assertEqual(meta.get("breadcrumbs"), ["Screen", "Test Settings"])
        self.assertEqual(body, "Body content here.")

    def test_no_frontmatter(self):
        text = "Just some content without frontmatter."
        meta, body = parse_frontmatter(text)
        self.assertEqual(meta, {})
        self.assertEqual(body, text.strip())

    def test_empty_frontmatter(self):
        text = "---\n---\nBody only."
        meta, body = parse_frontmatter(text)
        self.assertEqual(meta, {})
        self.assertEqual(body, "Body only.")

    def test_invalid_yaml_falls_back(self):
        text = "---\n: invalid yaml :\n---\nSome body."
        meta, body = parse_frontmatter(text)
        # Should fall back gracefully
        self.assertIsInstance(meta, dict)

    def test_source_url_extracted(self):
        text = "---\ntitle: My Article\nsource_url: https://support.example.com/articles/123\n---\ncontent"
        meta, _ = parse_frontmatter(text)
        self.assertEqual(meta.get("source_url"), "https://support.example.com/articles/123")

    def test_body_stripped(self):
        text = "---\ntitle: Test\n---\n\n  \n\nActual content starts here.\n"
        _, body = parse_frontmatter(text)
        self.assertTrue(body.startswith("Actual content starts here."))


class TestGetCompany(unittest.TestCase):

    def _path(self, relative: str) -> Path:
        """Create a fake path with the given relative structure."""
        return Path("/repo") / relative

    def test_hackerrank(self):
        p = self._path("data/hackerrank/screen/some-article.md")
        self.assertEqual(get_company(p), "hackerrank")

    def test_claude(self):
        p = self._path("data/claude/privacy-and-legal/some-article.md")
        self.assertEqual(get_company(p), "claude")

    def test_visa(self):
        p = self._path("data/visa/support/travellers-cheques.md")
        self.assertEqual(get_company(p), "visa")

    def test_unknown_path(self):
        p = self._path("other/stuff/file.md")
        self.assertEqual(get_company(p), "unknown")


class TestGetProductArea(unittest.TestCase):

    def _path(self, relative: str) -> Path:
        return Path("/repo") / relative

    def test_from_breadcrumbs_last_item(self):
        p = self._path("data/hackerrank/screen/test-settings.md")
        breadcrumbs = ["Screen", "Test Settings"]
        area = get_product_area(p, breadcrumbs)
        self.assertEqual(area, "test_settings")

    def test_from_path_when_no_breadcrumbs(self):
        p = self._path("data/hackerrank/interviews/interview-article.md")
        area = get_product_area(p, [])
        self.assertEqual(area, "interviews")

    def test_single_breadcrumb_uses_path(self):
        p = self._path("data/claude/privacy-and-legal/some-article.md")
        area = get_product_area(p, ["Claude"])  # only one breadcrumb — falls back to path
        # Hyphens normalized to underscores
        self.assertEqual(area, "privacy_and_legal")

    def test_hyphen_normalized(self):
        p = self._path("data/hackerrank/general-help/some-article.md")
        area = get_product_area(p, [])
        self.assertNotIn("-", area)  # should be normalized to underscore or kept as-is

    def test_empty_breadcrumbs_and_known_company(self):
        p = self._path("data/visa/support/article.md")
        area = get_product_area(p, [])
        self.assertEqual(area, "support")


if __name__ == "__main__":
    unittest.main(verbosity=2)
