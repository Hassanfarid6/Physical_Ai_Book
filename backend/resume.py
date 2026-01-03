"""
Resume functionality to continue from failure points.
Implements the ability to resume the ingestion pipeline from where it left off after a failure.
"""
import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime


class ResumeManager:
    """
    A class to manage the state of the ingestion pipeline to enable resumption
    from failure points.
    """
    
    def __init__(self, state_file: str = "pipeline_state.json"):
        """
        Initialize the resume manager.
        
        Args:
            state_file: Path to the file that stores pipeline state
        """
        self.state_file = state_file
        self.state_dir = Path(state_file).parent
        if not self.state_dir.exists():
            self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, 
                   processed_urls: List[str], 
                   failed_urls: List[str], 
                   current_step: str,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save the current state of the pipeline.
        
        Args:
            processed_urls: List of URLs that have been successfully processed
            failed_urls: List of URLs that failed to process
            current_step: Current step in the pipeline (e.g., 'crawling', 'extraction', 'embedding', 'storage')
            metadata: Additional metadata to save with the state
            
        Returns:
            True if state was saved successfully, False otherwise
        """
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "processed_urls": processed_urls,
                "failed_urls": failed_urls,
                "current_step": current_step,
                "metadata": metadata or {}
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Pipeline state saved to {self.state_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save pipeline state: {str(e)}")
            return False
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load the saved state of the pipeline.
        
        Returns:
            Dictionary containing the saved state, or None if no state exists
        """
        if not os.path.exists(self.state_file):
            logging.info(f"No saved state found at {self.state_file}")
            return None
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            logging.info(f"Pipeline state loaded from {self.state_file}")
            return state
            
        except Exception as e:
            logging.error(f"Failed to load pipeline state: {str(e)}")
            return None
    
    def get_remaining_urls(self, all_urls: List[str]) -> List[str]:
        """
        Get the URLs that still need to be processed based on saved state.
        
        Args:
            all_urls: List of all URLs that should be processed
            
        Returns:
            List of URLs that still need to be processed
        """
        state = self.load_state()
        if not state:
            return all_urls
        
        processed_urls = set(state.get("processed_urls", []))
        remaining_urls = [url for url in all_urls if url not in processed_urls]
        
        logging.info(f"Resuming pipeline: {len(remaining_urls)} URLs remaining to process")
        return remaining_urls
    
    def clear_state(self) -> bool:
        """
        Clear the saved state file.
        
        Returns:
            True if state was cleared successfully, False otherwise
        """
        try:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
                logging.info(f"Pipeline state cleared from {self.state_file}")
            return True
        except Exception as e:
            logging.error(f"Failed to clear pipeline state: {str(e)}")
            return False


# Global instance for convenience
resume_manager = ResumeManager()