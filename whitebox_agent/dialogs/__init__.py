"""
Whitebox AI Agent - Dialogs Module

Contains the UI components:
- ChatDockWidget: Main chat interface
- SettingsDockWidget: Settings panel
- UpdateCheckerDialog: Plugin update checker
"""

from .chat_dock import ChatDockWidget
from .settings_dock import SettingsDockWidget
from .update_checker import UpdateCheckerDialog

__all__ = [
    "ChatDockWidget",
    "SettingsDockWidget",
    "UpdateCheckerDialog",
]
