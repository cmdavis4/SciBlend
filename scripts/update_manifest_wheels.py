#!/usr/bin/env python3
"""
Update the wheels list in blender_manifest.toml based on downloaded wheels.
Usage: python3 scripts/update_manifest_wheels.py
"""

import os
from pathlib import Path
import re

def find_all_wheels(base_dir):
    """Find all .whl files in the wheels directory."""
    wheels_dir = Path(base_dir) / "wheels"

    if not wheels_dir.exists():
        print(f"Error: {wheels_dir} does not exist!")
        return []

    # Find all wheel files
    all_wheels = []

    # Common wheels (platform-independent)
    common_dir = wheels_dir / "common"
    if common_dir.exists():
        for whl in sorted(common_dir.glob("*.whl")):
            all_wheels.append(f"./wheels/common/{whl.name}")

    # Platform-specific wheels
    for platform in ["windows-x64", "linux-x64", "macos-x64", "macos-arm64"]:
        plat_dir = wheels_dir / platform
        if plat_dir.exists():
            for whl in sorted(plat_dir.glob("*.whl")):
                all_wheels.append(f"./wheels/{platform}/{whl.name}")

    return all_wheels

def update_manifest(base_dir, wheels):
    """Update blender_manifest.toml with the wheel list."""
    manifest_path = Path(base_dir) / "blender_manifest.toml"

    if not manifest_path.exists():
        print(f"Error: {manifest_path} does not exist!")
        return False

    # Read current manifest
    with open(manifest_path, 'r') as f:
        content = f.read()

    # Generate new wheels section
    wheels_section = "wheels = [\n"
    for wheel in wheels:
        wheels_section += f'  "{wheel}",\n'
    wheels_section += "]\n"

    # Replace wheels section using regex
    # Match from "wheels = [" to the next line that's just "]"
    pattern = r'wheels\s*=\s*\[[\s\S]*?\n\]'

    if re.search(pattern, content):
        new_content = re.sub(pattern, wheels_section.rstrip(), content)

        # Write updated manifest
        with open(manifest_path, 'w') as f:
            f.write(new_content)

        print(f"✓ Updated {manifest_path}")
        print(f"  Added {len(wheels)} wheel entries")
        return True
    else:
        print("Error: Could not find wheels section in manifest!")
        print("\nGenerated wheels section (copy manually):")
        print(wheels_section)
        return False

def main():
    # Get the project root directory (parent of scripts/)
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    print(f"Project directory: {project_dir}")

    # Find all wheels
    print("\nSearching for wheels...")
    wheels = find_all_wheels(project_dir)

    if not wheels:
        print("No wheels found! Run scripts/fetch_wheels.sh first.")
        return 1

    print(f"Found {len(wheels)} wheel files")

    # Show a sample
    print("\nSample wheels:")
    for wheel in wheels[:5]:
        print(f"  {wheel}")
    if len(wheels) > 5:
        print(f"  ... and {len(wheels) - 5} more")

    # Update manifest
    print("\nUpdating blender_manifest.toml...")
    success = update_manifest(project_dir, wheels)

    if success:
        print("\n✓ Success! blender_manifest.toml has been updated.")
        print("  Review the changes and test the extension.")
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())
