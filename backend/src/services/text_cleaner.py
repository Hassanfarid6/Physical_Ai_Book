import re
from typing import List
from backend.src.utils.logging import setup_logging

logger = setup_logging()


class TextCleaner:
    """
    Service to clean and normalize text content extracted from web pages.
    """
    
    def __init__(self):
        """Initialize the text cleaner."""
        pass
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and normalize line breaks
        text = self._normalize_whitespace(text)
        
        # Remove common web artifacts
        text = self._remove_web_artifacts(text)
        
        # Remove excessive newlines (more than 2 consecutive)
        text = re.sub(r'\\n{3,}', '\\n\\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text (spaces, tabs, newlines).
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Replace tabs and various space characters with single space
        text = re.sub(r'[\\t\\r\\f\\v]+', ' ', text)
        
        # Normalize different types of newlines
        text = re.sub(r'\\r\\n|\\r|\\n', '\\n', text)
        
        # Replace multiple spaces with single space (but preserve newlines)
        lines = text.split('\\n')
        normalized_lines = []
        
        for line in lines:
            # Replace multiple spaces with single space within each line
            line = re.sub(r' +', ' ', line)
            normalized_lines.append(line)
        
        return '\\n'.join(normalized_lines)
    
    def _remove_web_artifacts(self, text: str) -> str:
        """
        Remove common web artifacts from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with web artifacts removed
        """
        # Remove common navigation elements that might be picked up
        patterns_to_remove = [
            r'(?i)copyright\\s+\\d{4}.*?(?:\\n|$)',  # Copyright notices
            r'(?i)all\\s+rights\\s+reserved.*?(?:\\n|$)',  # Rights reserved
            r'(?i)privacy\\s+policy.*?(?:\\n|$)',  # Privacy policy links
            r'(?i)terms\\s+of\\s+service.*?(?:\\n|$)',  # Terms of service
            r'(?i)cookie\\s+policy.*?(?:\\n|$)',  # Cookie policy
            r'(?i)sitemap.*?(?:\\n|$)',  # Sitemap links
            r'(?i)home\\s+\\|\\s+.*?(?:\\n|$)',  # Navigation breadcrumbs
            r'(?i)next\\s+>.*?(?:\\n|$)',  # Next page links
            r'(?i)<\\s+previous.*?(?:\\n|$)',  # Previous page links
            r'(?i)jump\\s+to\\s+content.*?(?:\\n|$)',  # Skip navigation links
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text)
        
        # Remove multiple consecutive special characters that might be UI elements
        text = re.sub(r'[\\*\\-_=#]{10,}', '', text)
        
        # Remove URLs (but keep the text if it's meaningful)
        # This is a basic URL pattern; in practice, you might want a more robust solution
        text = re.sub(r'https?://[\\w\\.-]+\\.[\\w\\.-]+(?:/[\\w\\.-]*)*', '', text)
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving sentence boundaries.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        # Basic sentence splitting using common sentence terminators
        # This is a simple approach; for production use, consider using NLTK or spaCy
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up each sentence
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:  # Only add non-empty sentences
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences