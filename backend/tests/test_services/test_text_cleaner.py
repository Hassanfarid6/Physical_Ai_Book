import pytest
from backend.src.services.text_cleaner import TextCleaner


class TestTextCleaner:
    """Test cases for the TextCleaner service."""
    
    def test_initialization(self):
        """Test initializing TextCleaner."""
        cleaner = TextCleaner()
        assert cleaner is not None
    
    def test_clean_empty_text(self):
        """Test cleaning empty text."""
        cleaner = TextCleaner()
        result = cleaner.clean_text("")
        assert result == ""
    
    def test_clean_text_with_extra_whitespace(self):
        """Test cleaning text with extra whitespace."""
        cleaner = TextCleaner()
        dirty_text = "  This   has\t\textra   \\n\\nwhitespace  "
        expected = "This has extra\\n\\nwhitespace"
        result = cleaner.clean_text(dirty_text)
        assert result == expected
    
    def test_clean_text_with_web_artifacts(self):
        """Test cleaning text with web artifacts."""
        cleaner = TextCleaner()
        dirty_text = "Some content\\n\\nCopyright 2023 Example Corp\\nAll rights reserved"
        expected = "Some content"
        result = cleaner.clean_text(dirty_text)
        assert result == expected
    
    def test_clean_text_with_urls(self):
        """Test cleaning text with URLs."""
        cleaner = TextCleaner()
        dirty_text = "Visit https://example.com for more info"
        expected = "Visit  for more info"
        result = cleaner.clean_text(dirty_text)
        assert result == expected
    
    def test_split_into_sentences(self):
        """Test splitting text into sentences."""
        cleaner = TextCleaner()
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        sentences = cleaner.split_into_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0].strip() == "This is sentence one"
        assert sentences[1].strip() == "This is sentence two"
        assert sentences[2].strip() == "Is this sentence three"
    
    def test_normalize_whitespace(self):
        """Test normalizing whitespace in text."""
        cleaner = TextCleaner()
        text = "Text with\\ttabs,\\nnewlines,\\r\\nand\\rvarious\\fwhitespace"
        result = cleaner._normalize_whitespace(text)
        
        # Check that tabs and various newlines are normalized
        assert '\\t' not in result
        assert '\\r' not in result
        assert result.count('\\n') == 2  # Two newlines: one original, one from \\r\\n
    
    def test_remove_web_artifacts(self):
        """Test removing web artifacts from text."""
        cleaner = TextCleaner()
        text = "Content here\\nPrivacy Policy\\nTerms of Service\\nMore content"
        result = cleaner._remove_web_artifacts(text)
        
        assert "Privacy Policy" not in result
        assert "Terms of Service" not in result
        assert "Content here" in result
        assert "More content" in result