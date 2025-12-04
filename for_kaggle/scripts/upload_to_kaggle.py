#!/usr/bin/env python
"""
Upload the submission notebook and for_kaggle dataset to Kaggle automatically.

Requirements:
    - Kaggle API key configured (~/.kaggle/kaggle.json)
    - kaggle CLI installed

Usage:
    python scripts/upload_to_kaggle.py [--dataset-name NAME] [--notebook-name NAME]

This script will:
    1. Verify the for_kaggle.zip exists and passes model integrity tests
    2. Upload for_kaggle.zip as a Kaggle dataset
    3. Create/upload the submission notebook with the dataset attached
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, check=True, capture_output=False, error_msg=None):
    """Run a shell command and return result with detailed error feedback."""
    print(f"\n$ {' '.join(cmd)}")
    try:
        if capture_output:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr and not check:
                print(result.stderr, file=sys.stderr)
            return result
        else:
            return subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e:
        error_detail = ""
        if error_msg:
            error_detail = f"\n  {error_msg}"
        print(f"❌ Command failed with exit code {e.returncode}{error_detail}")
        if capture_output and e.stderr:
            print(f"\nError output:\n{e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError as e:
        print(f"❌ Command not found: {cmd[0]}")
        if error_msg:
            print(f"  {error_msg}")
        raise


def check_kaggle_cli():
    """Verify kaggle CLI is installed and configured."""
    try:
        run_command(['kaggle', '--version'], capture_output=True)
    except FileNotFoundError:
        print("❌ kaggle CLI not found")
        print("\nTo install: pip install kaggle")
        print("Docs: https://github.com/Kaggle/kaggle-api")
        sys.exit(1)
    
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print("❌ Kaggle API key not found")
        print(f"\nExpected location: {kaggle_json}")
        print("\nTo set up:")
        print("  1. Go to: https://www.kaggle.com/settings/account")
        print("  2. Click 'Create New Token' (downloads kaggle.json)")
        print(f"  3. Move it to: {kaggle_json}")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    
    print("✓ Kaggle CLI configured")


def get_kaggle_username():
    """Extract username from kaggle.json"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    with open(kaggle_json) as f:
        creds = json.load(f)
    return creds.get('username')


def upload_dataset(zip_path, dataset_name, username):
    """Upload for_kaggle.zip as a Kaggle dataset."""
    print(f"\n{'='*60}")
    print(f"UPLOADING DATASET")
    print(f"{'='*60}")
    
    # Generate comprehensive timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_compact = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dataset_id = f"{username}/{dataset_name}"
    
    # Check if dataset exists
    print(f"\nChecking if dataset {dataset_id} exists...")
    result = run_command(
        ['kaggle', 'datasets', 'list', '--mine'],
        check=False,
        capture_output=True,
        error_msg="Failed to list your datasets"
    )
    
    dataset_exists = dataset_name in result.stdout
    
    if dataset_exists:
        print(f"✓ Dataset {dataset_id} exists")
        # Create new version
        print(f"\nCreating new version of dataset...")
        version_msg = f'Updated {timestamp} (ID: {timestamp_compact})'
        try:
            run_command(
                ['kaggle', 'datasets', 'version', '-p', str(zip_path.parent), 
                 '-m', version_msg],
                error_msg=f"Failed to create new version of {dataset_id}"
            )
            print(f"✓ New version created")
            print(f"  - Message: {version_msg}")
        except subprocess.CalledProcessError:
            print(f"\n⚠️  Tip: Make sure you're in the dataset directory")
            print(f"   Dataset location: {zip_path.parent}")
            raise
    else:
        print(f"Dataset doesn't exist, creating new one...")
        # Create dataset.json metadata
        dataset_json = zip_path.parent / 'dataset-metadata.json'
        metadata = {
            'id': dataset_id,
            'licenses': [{'name': 'CC0-1.0'}],
            'resources': [
                {
                    'path': zip_path.name,
                    'description': 'NFL BDB 2026 inference model package with submission notebook'
                }
            ],
            'title': dataset_name,
            'subtitle': f'NFL position prediction model - {timestamp}',
            'keywords': ['nfl', 'big-data-bowl', 'prediction', 'inference'],
            'collab_sources': []
        }
        
        with open(dataset_json, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Created dataset metadata at {dataset_json}")
        print(f"  - Created: {timestamp}")
        print(f"  - Timestamp ID: {timestamp_compact}")
        
        try:
            run_command(
                ['kaggle', 'datasets', 'create', '-p', str(zip_path.parent)],
                error_msg=f"Failed to create dataset {dataset_id}"
            )
            print(f"✓ Dataset created successfully")
        except subprocess.CalledProcessError:
            print(f"\n⚠️  Troubleshooting:")
            print(f"  • Check dataset name: {dataset_name}")
            print(f"  • Must be lowercase alphanumeric with hyphens")
            print(f"  • Check that {zip_path} exists and is readable")
            raise
    
    print(f"✓ Dataset {dataset_id} ready")
    return dataset_id


def create_notebook(notebook_path, dataset_id, notebook_name, username):
    """Create a Kaggle notebook with the dataset attached."""
    print(f"\n{'='*60}")
    print(f"CREATING/UPLOADING NOTEBOOK")
    print(f"{'='*60}")
    
    # Read the notebook
    try:
        with open(notebook_path) as f:
            notebook_content = f.read()
        print(f"✓ Read notebook from {notebook_path}")
    except FileNotFoundError:
        print(f"❌ Notebook not found at {notebook_path}")
        raise
    
    # Generate comprehensive timestamp for tracking
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_compact = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a kernel metadata file
    kernel_json = notebook_path.parent / 'kernel-metadata.json'
    metadata = {
        'id': f'{username}/{notebook_name}',
        'title': notebook_name,
        'code_file': notebook_path.name,
        'language': 'python',
        'kernel_type': 'notebook',
        'dataset_sources': [dataset_id],
        'competition_sources': ['nfl-big-data-bowl-2026-prediction'],
        'kernel_integrations': [],
        'enable_gpu': False,
        'enable_internet': False,
        'dataset_license': None,
        'kernel_sources': [],
        'private': False,
        'enable_tpu': False,
        'docker_image_pinning_type': 'original'
    }
    
    try:
        with open(kernel_json, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Created notebook metadata at {kernel_json}")
        print(f"  - Dataset attached: {dataset_id}")
        print(f"  - Notebook ID: {metadata['id']}")
        print(f"  - Created: {timestamp}")
        print(f"  - Timestamp ID: {timestamp_compact}")
    except IOError as e:
        print(f"❌ Failed to write metadata file: {e}")
        raise
    
    return kernel_json


def main():
    parser = argparse.ArgumentParser(
        description='Upload NFL submission to Kaggle automatically',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/upload_to_kaggle.py
  python scripts/upload_to_kaggle.py --dataset-name nfl-model-v2 --notebook-name nfl-inference-v2
        """
    )
    parser.add_argument(
        '--dataset-name',
        default=f'nfl-model-v{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        help='Name of the Kaggle dataset to create/update (default: nfl-model-vYYYYMMDD-HHMMSS)'
    )
    parser.add_argument(
        '--notebook-name',
        default=f'nfl-inference-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        help='Name of the Kaggle notebook (default: nfl-inference-YYYYMMDD-HHMMSS)'
    )
    parser.add_argument(
        '--skip-dataset',
        action='store_true',
        help='Skip dataset upload, only create notebook metadata'
    )
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent
    zip_path = repo_root / 'for_kaggle.zip'
    notebook_path = repo_root / 'notebooks' / 'submission_notebook.ipynb'
    
    # Verify files exist
    print(f"\n{'='*60}")
    print(f"VERIFICATION")
    print(f"{'='*60}")
    
    errors = []
    
    if not zip_path.exists():
        error = f"for_kaggle.zip not found at {zip_path}"
        print(f"❌ {error}")
        print(f"   Fix: Run 'make prepare-for-kaggle' first")
        errors.append(error)
    else:
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"✓ Found {zip_path.name} ({size_mb:.1f} MB)")
    
    if not notebook_path.exists():
        error = f"submission_notebook.ipynb not found at {notebook_path}"
        print(f"❌ {error}")
        errors.append(error)
    else:
        print(f"✓ Found {notebook_path.name}")
    
    if errors:
        print(f"\n❌ Verification failed with {len(errors)} error(s)")
        sys.exit(1)
    
    try:
        check_kaggle_cli()
        username = get_kaggle_username()
        print(f"✓ Kaggle username: {username}")
    except SystemExit:
        raise
    except Exception as e:
        print(f"❌ Failed to verify Kaggle setup: {e}")
        sys.exit(1)
    
    try:
        if not args.skip_dataset:
            dataset_id = upload_dataset(zip_path, args.dataset_name, username)
        else:
            dataset_id = f"{username}/{args.dataset_name}"
            print(f"\n⏭️  Skipping dataset upload")
        
        kernel_json = create_notebook(notebook_path, dataset_id, args.notebook_name, username)
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS - READY TO PUSH")
        print(f"{'='*60}")
        print(f"\nNotebook metadata created:")
        print(f"  Location: {kernel_json}")
        print(f"  Dataset: {dataset_id}")
        print(f"\nTo push the notebook to Kaggle, run:")
        print(f"  cd notebooks")
        print(f"  kaggle kernels push")
        print(f"\nThen:")
        print(f"  1. Go to https://www.kaggle.com/code/[your-username]/[notebook-name]")
        print(f"  2. Click 'Commit and run'")
        print(f"  3. Enable 'Submit to Competition' option")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ UPLOAD FAILED")
        print(f"{'='*60}")
        print(f"\nThe upload script encountered an error.")
        print(f"Exit code: {e.returncode}")
        print(f"\nCommon issues:")
        print(f"  • Dataset name has invalid characters (use lowercase, hyphens only)")
        print(f"  • Not enough permissions on Kaggle account")
        print(f"  • API key is outdated or invalid")
        print(f"\nTroubleshooting:")
        print(f"  1. Check your Kaggle API key: ~/.kaggle/kaggle.json")
        print(f"  2. Regenerate if needed: https://www.kaggle.com/settings/account")
        print(f"  3. Verify for_kaggle.zip exists and is readable")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR")
        print(f"{'='*60}")
        print(f"\n{e}")
        print(f"\nPlease report this error with the full output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
