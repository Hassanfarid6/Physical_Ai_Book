"""
Verification script to confirm successful pipeline execution.
This script checks that the pipeline ran correctly and produced expected outputs.
"""
import os
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
from config import Config
from qdrant_client import QdrantClient


def verify_environment():
    """Verify that all required environment variables are set."""
    print("Verifying environment configuration...")
    
    config_errors = Config.validate()
    if config_errors:
        print("❌ Configuration errors found:")
        for error in config_errors:
            print(f"  - {error}")
        return False
    
    print("✅ All required environment variables are set")
    return True


def verify_temp_storage():
    """Verify that temporary storage was created and contains data."""
    print("\nVerifying temporary storage...")
    
    temp_dir = Path("temp_storage")
    if not temp_dir.exists():
        print("⚠️  Temporary storage directory does not exist")
        return False
    
    files = list(temp_dir.glob("*.json"))
    if not files:
        print("⚠️  No temporary storage files found")
        return False
    
    print(f"✅ Found {len(files)} temporary storage files")
    
    # Check the most recent file
    recent_file = max(files, key=lambda x: x.stat().st_mtime)
    print(f"  Most recent file: {recent_file.name}")
    
    try:
        with open(recent_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  Contains {len(data)} items")
    except Exception as e:
        print(f"⚠️  Could not read temporary file: {e}")
        return False
    
    return True


def verify_qdrant_storage():
    """Verify that embeddings were stored in Qdrant."""
    print("\nVerifying Qdrant storage...")
    
    try:
        client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=10
        )
        
        # Get collection info
        collection_info = client.get_collection(Config.QDRANT_COLLECTION_NAME)
        
        print(f"✅ Successfully connected to Qdrant collection '{Config.QDRANT_COLLECTION_NAME}'")
        print(f"  Vector count: {collection_info.points_count}")
        print(f"  Vector size: {collection_info.config.params.vectors.size}")
        
        if collection_info.points_count > 0:
            print("✅ Embeddings were successfully stored in Qdrant")
            return True
        else:
            print("⚠️  Qdrant collection exists but is empty")
            return False
            
    except Exception as e:
        print(f"❌ Could not verify Qdrant storage: {e}")
        return False


def verify_pipeline_execution():
    """Verify that the pipeline executed successfully."""
    print("\nVerifying pipeline execution...")
    
    # Check for log file
    log_file = Path("ingestion_pipeline.log")
    if log_file.exists():
        print("✅ Log file exists")
        
        # Check if the log contains successful completion messages
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        if "Pipeline execution completed" in log_content:
            print("✅ Pipeline execution completed successfully")
            return True
        else:
            print("⚠️  Pipeline may not have completed successfully")
            return False
    else:
        print("⚠️  Log file does not exist")
        return False


def main():
    """Main verification function."""
    print("Running verification checks for Docusaurus Ingestion Pipeline...")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Run all verification checks
    checks = [
        ("Environment Configuration", verify_environment),
        ("Temporary Storage", verify_temp_storage),
        ("Qdrant Storage", verify_qdrant_storage),
        ("Pipeline Execution", verify_pipeline_execution),
    ]
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * len(check_name))
        if not check_func():
            all_checks_passed = False
    
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("🎉 All verification checks passed! The pipeline executed successfully.")
        return 0
    else:
        print("❌ Some verification checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())