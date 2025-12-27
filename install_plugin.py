#!/usr/bin/env python3
"""
Whitebox AI Agent - Plugin Installation Script

This script installs the plugin to the QGIS plugin directory.
Run this script to install or update the plugin.

Usage:
    python install_plugin.py

The script will:
1. Detect the QGIS plugin directory
2. Copy the plugin files to the directory
3. Report success or failure
"""

import shutil
import sys
from pathlib import Path


def get_qgis_plugin_dir() -> Path:
    """Get the QGIS plugin directory based on OS."""
    home = Path.home()

    if sys.platform == "darwin":
        # macOS
        return (
            home
            / "Library"
            / "Application Support"
            / "QGIS"
            / "QGIS3"
            / "profiles"
            / "default"
            / "python"
            / "plugins"
        )
    elif sys.platform == "win32":
        # Windows
        return (
            home
            / "AppData"
            / "Roaming"
            / "QGIS"
            / "QGIS3"
            / "profiles"
            / "default"
            / "python"
            / "plugins"
        )
    else:
        # Linux
        return (
            home
            / ".local"
            / "share"
            / "QGIS"
            / "QGIS3"
            / "profiles"
            / "default"
            / "python"
            / "plugins"
        )


def install_plugin():
    """Install the plugin to QGIS."""
    # Get paths
    script_dir = Path(__file__).parent
    plugin_source = script_dir / "whitebox_agent"
    plugin_dir = get_qgis_plugin_dir()
    plugin_dest = plugin_dir / "whitebox_agent"

    print("=" * 60)
    print("Whitebox AI Agent - Plugin Installer")
    print("=" * 60)
    print()

    # Check source exists
    if not plugin_source.exists():
        print(f"ERROR: Plugin source not found at {plugin_source}")
        sys.exit(1)

    print(f"Source: {plugin_source}")
    print(f"Destination: {plugin_dest}")
    print()

    # Create plugin directory if needed
    if not plugin_dir.exists():
        print(f"Creating plugin directory: {plugin_dir}")
        plugin_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing installation
    if plugin_dest.exists():
        print("Removing existing installation...")
        shutil.rmtree(plugin_dest)

    # Copy plugin
    print("Copying plugin files...")
    shutil.copytree(plugin_source, plugin_dest)

    # Verify installation
    init_file = plugin_dest / "__init__.py"
    if init_file.exists():
        print()
        print("=" * 60)
        print("✅ Installation successful!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Restart QGIS (if running)")
        print("2. Go to Plugins → Manage and Install Plugins")
        print("3. Enable 'Whitebox AI Agent'")
        print("4. The plugin will appear in the toolbar")
        print()
        print("Requirements:")
        print("- WhiteboxTools Processing provider must be installed")
        print("- Configure your LLM provider in the Settings panel")
        print()
    else:
        print()
        print("❌ Installation may have failed. Please check the paths.")
        sys.exit(1)


if __name__ == "__main__":
    install_plugin()
