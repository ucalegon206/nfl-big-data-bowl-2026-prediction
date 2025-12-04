#!/usr/bin/env python
"""
cleanup_old_datasets.py

Automated cleanup of old cached datasets from Kaggle.
This script identifies and lists old datasets that should be deleted,
then optionally deletes them using the Kaggle API.

WARNING: This script uses the Kaggle API to delete datasets. Use with caution.

Usage:
    python scripts/cleanup_old_datasets.py [--list-only] [--delete-old]

Options:
    --list-only     Only list old datasets, don't delete them (default)
    --delete-old    Automatically delete old datasets (requires confirmation)
    --help          Show this help message
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command and return result."""
    try:
        if capture_output:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result
        else:
            return subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        if capture_output and e.stderr:
            print(f"Error: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return None


def get_kaggle_username():
    """Extract username from kaggle.json"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    try:
        with open(kaggle_json) as f:
            config = json.load(f)
            return config.get('username', 'unknown')
    except Exception:
        return 'unknown'


def list_kaggle_datasets():
    """List all user's Kaggle datasets using CLI."""
    username = get_kaggle_username()
    
    print(f"\n🔍 Fetching datasets for user: {username}\n")
    
    # Use kaggle CLI to list datasets
    result = run_command(
        ['kaggle', 'datasets', 'list', '-s', 'nfl-model'],
        capture_output=True
    )
    
    if not result or result.returncode != 0:
        print("❌ Failed to list datasets")
        print("\nTroubleshooting:")
        print("  1. Verify Kaggle API key: ~/.kaggle/kaggle.json")
        print("  2. Check Kaggle CLI: kaggle --version")
        print("  3. Test connection: kaggle competitions list")
        return []
    
    lines = result.stdout.strip().split('\n')
    if len(lines) < 2:
        print("No datasets found")
        return []
    
    # Parse dataset list output
    # Expected format: ref | size | downloadCount | lastUpdated
    datasets = []
    for line in lines[1:]:  # Skip header
        if line.strip():
            parts = line.split('|')
            if len(parts) >= 1:
                ref = parts[0].strip()
                datasets.append({
                    'ref': ref,
                    'raw_line': line.strip()
                })
    
    return datasets


def identify_old_datasets(datasets):
    """Identify which datasets are old (should be deleted)."""
    old_datasets = []
    new_datasets = []
    
    for dataset in datasets:
        ref = dataset['ref']
        
        # Patterns:
        # OLD: nfl-model-v20251204 (no time component)
        # OLD: nfl-model-v20251204-100817 (has time but in old format)
        # NEW: nfl-model-v20251204-101918 (YYYYMMDD-HHMMSS format with seconds)
        
        # Check if it has the full timestamp pattern (YYYYMMDD-HHMMSS with exactly 6 digits after dash)
        if 'nfl-model-v' in ref:
            parts = ref.split('-')
            # NEW format should be: nfl, model, v20251204, HHMMSS (and possibly more parts)
            if len(parts) >= 4:
                # Check if last part before potential other segments is 6 digits (HHMMSS)
                time_part = parts[3] if len(parts) > 3 else ''
                if time_part.isdigit() and len(time_part) == 6:
                    # This is new format (has HHMMSS)
                    new_datasets.append(dataset)
                else:
                    # This is old format
                    old_datasets.append(dataset)
            else:
                # Not enough parts, likely old format
                old_datasets.append(dataset)
    
    return old_datasets, new_datasets


def delete_dataset(dataset_ref):
    """Delete a dataset using Kaggle CLI."""
    print(f"  Deleting: {dataset_ref}...", end=' ')
    result = run_command(
        ['kaggle', 'datasets', 'delete', '-p', dataset_ref],
        capture_output=True
    )
    
    if result and result.returncode == 0:
        print("✓")
        return True
    else:
        print("✗")
        if result and result.stderr:
            print(f"    Error: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Clean up old cached datasets from Kaggle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cleanup_old_datasets.py --list-only
      List old datasets (default behavior, no deletion)
  
  python scripts/cleanup_old_datasets.py --delete-old
      Delete old datasets after confirmation
        """
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        default=False,
        help='Only list old datasets, do not delete (default)'
    )
    parser.add_argument(
        '--delete-old',
        action='store_true',
        help='Automatically delete old datasets (requires confirmation)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Kaggle Dataset Cleanup Tool")
    print("="*70)
    
    # List all datasets
    datasets = list_kaggle_datasets()
    
    if not datasets:
        print("❌ Could not retrieve datasets")
        sys.exit(1)
    
    print(f"Total datasets found: {len(datasets)}\n")
    
    # Identify old vs new
    old_datasets, new_datasets = identify_old_datasets(datasets)
    
    print(f"📊 Breakdown:")
    print(f"  • Old datasets (should delete): {len(old_datasets)}")
    print(f"  • New datasets (keep): {len(new_datasets)}\n")
    
    if old_datasets:
        print("❌ OLD DATASETS (candidates for deletion):")
        for dataset in old_datasets:
            print(f"  • {dataset['ref']}")
        print()
    
    if new_datasets:
        print("✓ NEW DATASETS (these should be kept):")
        for dataset in new_datasets:
            print(f"  • {dataset['ref']}")
        print()
    
    if not old_datasets:
        print("✓ No old datasets found. Your Kaggle datasets are clean!")
        return
    
    # Deletion handling
    if args.delete_old:
        print(f"⚠️  WARNING: About to delete {len(old_datasets)} dataset(s)")
        print("\nDatasets to delete:")
        for dataset in old_datasets:
            print(f"  • {dataset['ref']}")
        print()
        
        response = input("Type 'yes' to confirm deletion: ").strip().lower()
        
        if response != 'yes':
            print("❌ Deletion cancelled")
            return
        
        print(f"\n🗑️  Deleting {len(old_datasets)} old dataset(s)...\n")
        
        deleted = 0
        failed = 0
        for dataset in old_datasets:
            if delete_dataset(dataset['ref']):
                deleted += 1
            else:
                failed += 1
        
        print(f"\n✓ Deletion complete: {deleted} deleted, {failed} failed")
        
        if failed == 0:
            print("✓ All old datasets cleaned up!")
        else:
            print(f"⚠️  {failed} dataset(s) failed to delete. Try manually:")
            print("   https://www.kaggle.com/settings/datasets")
    else:
        print("ℹ️  Use --delete-old flag to automatically delete these datasets")
        print("   Run: python scripts/cleanup_old_datasets.py --delete-old")


if __name__ == '__main__':
    main()
