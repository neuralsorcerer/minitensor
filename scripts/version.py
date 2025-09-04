# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Version management script for minitensor.

This script helps manage version numbers across the project files.
"""

import argparse
import re
import sys
from pathlib import Path


class VersionManager:
    """Manages version numbers across project files."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.version_files = {
            "pyproject.toml": self._update_pyproject_version,
            "Cargo.toml": self._update_cargo_version,
            "engine/Cargo.toml": self._update_cargo_version,
            "bindings/Cargo.toml": self._update_cargo_version,
        }

    def get_current_version(self) -> str:
        """Get the current version from pyproject.toml."""
        pyproject_path = self.project_root / "pyproject.toml"

        if not pyproject_path.exists():
            return "0.1.0"

        content = pyproject_path.read_text()

        # Try to find version in dynamic field first
        if 'dynamic = ["version"]' in content:
            # Version is managed by setuptools_scm, try to get from git
            try:
                import subprocess

                result = subprocess.run(
                    ["git", "describe", "--tags", "--abbrev=0"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                )
                if result.returncode == 0:
                    return result.stdout.strip().lstrip("v")
            except:
                pass

        # Fallback to looking for version field
        version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if version_match:
            return version_match.group(1)

        return "0.1.0"

    def _update_pyproject_version(self, content: str, new_version: str) -> str:
        """Update version in pyproject.toml."""
        # If using dynamic versioning, don't change it
        if 'dynamic = ["version"]' in content:
            return content

        # Update version field
        pattern = r'(version\s*=\s*)"[^"]+"'
        replacement = f'\\1"{new_version}"'
        return re.sub(pattern, replacement, content)

    def _update_cargo_version(self, content: str, new_version: str) -> str:
        """Update version in Cargo.toml."""
        pattern = r'(version\s*=\s*)"[^"]+"'
        replacement = f'\\1"{new_version}"'
        return re.sub(pattern, replacement, content)

    def update_version(self, new_version: str) -> None:
        """Update version in all relevant files."""
        print(f"Updating version to {new_version}")

        for file_path, update_func in self.version_files.items():
            full_path = self.project_root / file_path

            if not full_path.exists():
                print(f"Warning: {file_path} not found, skipping")
                continue

            try:
                content = full_path.read_text()
                updated_content = update_func(content, new_version)

                if content != updated_content:
                    full_path.write_text(updated_content)
                    print(f"Updated {file_path}")
                else:
                    print(f"No changes needed in {file_path}")

            except Exception as e:
                print(f"Error updating {file_path}: {e}")

    def validate_version(self, version: str) -> bool:
        """Validate version string format."""
        pattern = r"^\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?$"
        return bool(re.match(pattern, version))

    def bump_version(self, part: str) -> str:
        """Bump version part (major, minor, patch)."""
        current = self.get_current_version()

        # Parse current version
        version_parts = current.split(".")
        if len(version_parts) < 3:
            version_parts.extend(["0"] * (3 - len(version_parts)))

        major, minor, patch = map(int, version_parts[:3])

        if part == "major":
            major += 1
            minor = 0
            patch = 0
        elif part == "minor":
            minor += 1
            patch = 0
        elif part == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid version part: {part}")

        return f"{major}.{minor}.{patch}"

    def create_git_tag(self, version: str) -> None:
        """Create a git tag for the version."""
        try:
            import subprocess

            tag_name = f"v{version}"

            # Create tag
            subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", f"Release {version}"],
                check=True,
                cwd=self.project_root,
            )

            print(f"Created git tag: {tag_name}")

        except subprocess.CalledProcessError as e:
            print(f"Error creating git tag: {e}")
        except ImportError:
            print("Git not available, skipping tag creation")


def main():
    parser = argparse.ArgumentParser(description="Manage minitensor version")
    parser.add_argument("--current", action="store_true", help="Show current version")
    parser.add_argument("--set", metavar="VERSION", help="Set specific version")
    parser.add_argument(
        "--bump", choices=["major", "minor", "patch"], help="Bump version part"
    )
    parser.add_argument(
        "--tag", action="store_true", help="Create git tag after version update"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    version_manager = VersionManager(project_root)

    if args.current:
        print(version_manager.get_current_version())
        return

    new_version = None

    if args.set:
        if not version_manager.validate_version(args.set):
            print(f"Invalid version format: {args.set}")
            print("Expected format: X.Y.Z or X.Y.Z-alpha.N")
            sys.exit(1)
        new_version = args.set

    elif args.bump:
        new_version = version_manager.bump_version(args.bump)

    else:
        parser.print_help()
        return

    if new_version:
        version_manager.update_version(new_version)

        if args.tag:
            version_manager.create_git_tag(new_version)

        print(f"Version updated to {new_version}")


if __name__ == "__main__":
    main()
