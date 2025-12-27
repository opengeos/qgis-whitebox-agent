"""
Chat Dock Widget

Main chat-style interface for the Whitebox AI Agent.
Provides a conversational UI for natural language interaction with WhiteboxTools.
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal, QSettings
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QProgressBar,
    QScrollArea,
    QFrame,
    QSplitter,
    QMessageBox,
)
from qgis.PyQt.QtGui import QFont, QColor, QTextCursor, QTextCharFormat
from qgis.core import QgsProject

from ..core.processing_registry import ProcessingRegistryAdapter
from ..core.context_builder import ContextBuilder
from ..core.llm_client import LLMClient, LLMConfig, LLMProvider
from ..core.agent_executor import AgentExecutor, ActionType


class AgentWorker(QThread):
    """Background worker for LLM communication and execution."""

    finished = pyqtSignal(dict)  # Emits result dictionary
    error = pyqtSignal(str)  # Emits error message
    progress = pyqtSignal(float)  # Emits progress (0-100)
    message = pyqtSignal(str)  # Emits status messages

    def __init__(
        self,
        llm_client: LLMClient,
        context_builder: ContextBuilder,
        agent_executor: AgentExecutor,
        user_message: str,
        conversation_history: List[Dict[str, str]],
    ):
        super().__init__()
        self.llm_client = llm_client
        self.context_builder = context_builder
        self.agent_executor = agent_executor
        self.user_message = user_message
        self.conversation_history = conversation_history

    def run(self):
        """Execute the agent workflow in background."""
        try:
            self.message.emit("Building context...")

            # Build context
            context = self.context_builder.build_full_context(
                include_algorithms=True,
                include_layers=True,
            )

            self.message.emit("Sending to LLM...")
            self.progress.emit(25)

            # Send to LLM
            response = self.llm_client.send_message(
                user_message=self.user_message,
                context=context,
                conversation_history=self.conversation_history,
            )

            self.progress.emit(50)
            self.message.emit("Processing response...")

            # Check for errors in parsing
            if response.get("error"):
                self.finished.emit(
                    {
                        "success": False,
                        "action_type": "explain",
                        "message": response.get("text", "Failed to parse LLM response"),
                        "raw_response": response,
                    }
                )
                return

            # Execute the action
            result = self.agent_executor.execute(
                response,
                progress_callback=lambda p: self.progress.emit(50 + p * 0.5),
            )

            self.progress.emit(100)

            self.finished.emit(
                {
                    "success": result.success,
                    "action_type": result.action_type.value,
                    "message": result.message,
                    "data": result.data,
                    "outputs": result.outputs,
                    "error": result.error,
                    "raw_response": response,
                }
            )

        except Exception as e:
            import traceback

            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class ChatDockWidget(QDockWidget):
    """Main chat dock widget for the Whitebox AI Agent."""

    SETTINGS_PREFIX = "WhiteboxAgent/"

    def __init__(self, iface, parent=None):
        """Initialize the chat dock widget.

        Args:
            iface: QGIS interface instance.
            parent: Parent widget.
        """
        super().__init__("Whitebox AI Agent", parent)
        self.iface = iface
        self.settings = QSettings()

        # Initialize core components
        self.registry_adapter = ProcessingRegistryAdapter()
        self.context_builder = ContextBuilder(self.registry_adapter)
        self.agent_executor = AgentExecutor(self.registry_adapter, self.context_builder)

        # LLM client (initialized on first use)
        self._llm_client: Optional[LLMClient] = None

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Worker thread
        self._worker: Optional[AgentWorker] = None

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Set up the dock widget UI."""
        main_widget = QWidget()
        self.setWidget(main_widget)

        layout = QVBoxLayout(main_widget)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header_layout = QHBoxLayout()

        header_label = QLabel("Whitebox AI Agent")
        header_font = QFont()
        header_font.setPointSize(11)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        # LLM Provider selector
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Ollama", "ollama")
        self.provider_combo.addItem("Claude", "claude")
        self.provider_combo.addItem("OpenAI", "openai")
        self.provider_combo.addItem("Gemini", "gemini")
        self.provider_combo.addItem("Bedrock", "bedrock")
        self.provider_combo.setMinimumWidth(100)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        header_layout.addWidget(QLabel("Provider:"))
        header_layout.addWidget(self.provider_combo)

        layout.addLayout(header_layout)

        # Chat history display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(300)
        self.chat_display.setStyleSheet(
            """
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10pt;
            }
        """
        )
        layout.addWidget(self.chat_display, stretch=1)

        # Status bar
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888; font-size: 9pt;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        # Algorithm count
        alg_count = len(self.registry_adapter.get_whitebox_algorithms())
        self.alg_count_label = QLabel(f"{alg_count} algorithms")
        self.alg_count_label.setStyleSheet("color: #888; font-size: 9pt;")
        status_layout.addWidget(self.alg_count_label)

        layout.addLayout(status_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(8)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Input area
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(
            "Ask me to analyze your data... (e.g., 'Fill sinks in my DEM')"
        )
        self.input_field.setStyleSheet(
            """
            QLineEdit {
                padding: 8px;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                font-size: 10pt;
            }
            QLineEdit:focus {
                border-color: #0078d4;
            }
        """
        )
        self.input_field.returnPressed.connect(self._on_send)
        input_layout.addWidget(self.input_field)

        self.send_btn = QPushButton("Send")
        self.send_btn.setMinimumWidth(60)
        self.send_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbd;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """
        )
        self.send_btn.clicked.connect(self._on_send)
        input_layout.addWidget(self.send_btn)

        layout.addLayout(input_layout)

        # Action buttons
        button_layout = QHBoxLayout()

        self.clear_btn = QPushButton("Clear Chat")
        self.clear_btn.clicked.connect(self._clear_chat)
        button_layout.addWidget(self.clear_btn)

        self.list_alg_btn = QPushButton("List Algorithms")
        self.list_alg_btn.clicked.connect(self._list_algorithms)
        button_layout.addWidget(self.list_alg_btn)

        self.list_layers_btn = QPushButton("Show Layers")
        self.list_layers_btn.clicked.connect(self._show_layers)
        button_layout.addWidget(self.list_layers_btn)

        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Welcome message
        self._add_system_message(
            "Welcome to Whitebox AI Agent!\n\n"
            "I can help you run WhiteboxTools algorithms through natural language.\n\n"
            "Try asking me things like:\n"
            "  • 'Fill depressions in my DEM'\n"
            "  • 'Calculate flow accumulation'\n"
            "  • 'Extract streams from the DEM'\n"
            "  • 'What algorithms are available for hydrology?'\n\n"
            "Make sure you have layers loaded in QGIS first!"
        )

    def _get_llm_client(self) -> Optional[LLMClient]:
        """Get or create the LLM client based on current settings."""
        provider = self.provider_combo.currentData()

        # Get settings based on provider
        if provider == "bedrock":
            # Bedrock uses AWS credentials, not API key
            api_key = None
            model = self.settings.value(f"{self.SETTINGS_PREFIX}bedrock_model", "")
            base_url = self.settings.value(f"{self.SETTINGS_PREFIX}bedrock_region", "")
        else:
            api_key = self.settings.value(
                f"{self.SETTINGS_PREFIX}{provider}_api_key", ""
            )
            model = self.settings.value(f"{self.SETTINGS_PREFIX}{provider}_model", "")
            base_url = self.settings.value(
                f"{self.SETTINGS_PREFIX}{provider}_base_url", ""
            )

        # For Ollama and Bedrock, no API key needed
        if provider in ("ollama", "bedrock"):
            api_key = None
        elif not api_key:
            self._add_system_message(
                f"[Warning] No API key configured for {provider}. "
                f"Please configure it in Settings."
            )
            return None

        config = LLMClient.create_config(
            provider=provider,
            api_key=api_key or None,
            model=model or None,
            base_url=base_url or None,
        )

        return LLMClient(config)

    def _on_provider_changed(self, index: int):
        """Handle provider selection change."""
        provider = self.provider_combo.currentData()
        self.settings.setValue(f"{self.SETTINGS_PREFIX}provider", provider)
        self._llm_client = None  # Reset client
        self._add_system_message(
            f"Switched to {self.provider_combo.currentText()} provider."
        )

    def _on_send(self):
        """Handle send button click."""
        user_message = self.input_field.text().strip()
        if not user_message:
            return

        # Clear input
        self.input_field.clear()

        # Add user message to chat
        self._add_user_message(user_message)

        # Get LLM client
        llm_client = self._get_llm_client()
        if not llm_client:
            return

        # Disable input while processing
        self._set_processing_state(True)

        # Create and start worker
        self._worker = AgentWorker(
            llm_client=llm_client,
            context_builder=self.context_builder,
            agent_executor=self.agent_executor,
            user_message=user_message,
            conversation_history=self.conversation_history.copy(),
        )

        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.message.connect(self._on_worker_message)

        self._worker.start()

        # Add to conversation history
        self.conversation_history.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

    def _on_worker_finished(self, result: Dict[str, Any]):
        """Handle worker completion."""
        self._set_processing_state(False)

        # Debug: log full result
        print(f"[WhiteboxAgent] Worker finished with result: {result}")

        action_type = result.get("action_type", "explain")
        message = result.get("message", "No response")
        success = result.get("success", False)

        # Format the response based on action type
        if action_type == "ask_user":
            self._add_assistant_message(f"[Question] {message}")
        elif action_type == "select_algorithm":
            self._add_assistant_message(f"[Suggestions]\n{message}")
        elif action_type == "run_algorithm":
            if success:
                # Load outputs on main thread (required by QGIS)
                loaded = []
                outputs = result.get("outputs", {})
                should_load = result.get("data", {}).get("load_outputs", True)

                print(f"[WhiteboxAgent] Outputs to load: {outputs}")
                print(f"[WhiteboxAgent] Should load: {should_load}")

                if should_load and outputs:
                    loaded = self._load_output_layers(outputs)

                print(f"[WhiteboxAgent] Loaded layers: {loaded}")

                if loaded:
                    self._add_assistant_message(
                        f"[OK] {message}\n\nLoaded layers: {', '.join(loaded)}"
                    )
                else:
                    self._add_assistant_message(f"[OK] {message}")
            else:
                self._add_assistant_message(f"[Error] {message}")
        elif action_type == "explain":
            self._add_assistant_message(f"{message}")
        else:
            self._add_assistant_message(message)

        # Add to conversation history
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": message,
            }
        )

        # Limit history size
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _on_worker_error(self, error_message: str):
        """Handle worker error."""
        self._set_processing_state(False)
        self._add_system_message(f"[Error] {error_message}")

    def _load_output_layers(self, outputs: Dict[str, Any]) -> List[str]:
        """
        Load output layers into QGIS project (must be called from main thread).

        Args:
            outputs: Dictionary of output names to paths/values.

        Returns:
            List of loaded layer names.
        """
        import os
        from qgis.core import QgsRasterLayer, QgsVectorLayer, QgsProject

        loaded = []
        project = QgsProject.instance()

        print(f"[WhiteboxAgent] Loading outputs on main thread: {outputs}")

        for output_name, output_value in outputs.items():
            print(f"[WhiteboxAgent] Checking output: {output_name} = {output_value}")

            if output_value is None:
                continue

            # Skip non-path outputs
            if not isinstance(output_value, str):
                print(f"[WhiteboxAgent] Skipping {output_name}: not a string")
                continue

            # Check if it's a file path
            if not os.path.exists(output_value):
                print(f"[WhiteboxAgent] File not found: {output_value}")
                continue

            # Determine layer name
            layer_name = os.path.splitext(os.path.basename(output_value))[0]
            print(
                f"[WhiteboxAgent] Attempting to load: {output_value} as '{layer_name}'"
            )

            # Try as raster first (common for WhiteboxTools)
            layer = QgsRasterLayer(output_value, layer_name)
            if layer.isValid():
                project.addMapLayer(layer)
                loaded.append(layer_name)
                print(f"[WhiteboxAgent] Loaded raster: {layer_name}")
                continue

            # Try as vector
            layer = QgsVectorLayer(output_value, layer_name, "ogr")
            if layer.isValid():
                project.addMapLayer(layer)
                loaded.append(layer_name)
                print(f"[WhiteboxAgent] Loaded vector: {layer_name}")
                continue

            print(f"[WhiteboxAgent] Could not load as raster or vector: {output_value}")

        print(f"[WhiteboxAgent] Total loaded: {loaded}")
        return loaded

    def _on_worker_progress(self, progress: float):
        """Handle progress updates."""
        self.progress_bar.setValue(int(progress))

    def _on_worker_message(self, message: str):
        """Handle status message updates."""
        self.status_label.setText(message)

    def _set_processing_state(self, processing: bool):
        """Set UI state during processing."""
        self.send_btn.setEnabled(not processing)
        self.input_field.setEnabled(not processing)
        self.progress_bar.setVisible(processing)
        self.progress_bar.setValue(0)

        if processing:
            self.status_label.setText("Processing...")
        else:
            self.status_label.setText("Ready")

    def _add_user_message(self, message: str):
        """Add a user message to the chat display."""
        timestamp = datetime.now().strftime("%H:%M")
        self._append_formatted_message(f"[{timestamp}] You:", message, "#569cd6")

    def _add_assistant_message(self, message: str):
        """Add an assistant message to the chat display."""
        timestamp = datetime.now().strftime("%H:%M")
        self._append_formatted_message(f"[{timestamp}] Agent:", message, "#4ec9b0")

    def _add_system_message(self, message: str):
        """Add a system message to the chat display."""
        self._append_formatted_message("", message, "#ce9178")

    def _append_formatted_message(self, header: str, body: str, header_color: str):
        """Append a formatted message to the chat display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Add spacing if not first message
        if cursor.position() > 0:
            cursor.insertText("\n\n")

        # Add header with color
        if header:
            header_format = QTextCharFormat()
            header_format.setForeground(QColor(header_color))
            header_format.setFontWeight(QFont.Bold)
            cursor.insertText(header + "\n", header_format)

        # Add body
        body_format = QTextCharFormat()
        body_format.setForeground(QColor("#d4d4d4"))
        cursor.insertText(body, body_format)

        # Scroll to bottom
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _clear_chat(self):
        """Clear the chat history."""
        self.chat_display.clear()
        self.conversation_history.clear()
        self._add_system_message("Chat cleared. Ready for new conversation.")

    def _list_algorithms(self):
        """List available WhiteboxTools algorithms."""
        algorithms = self.registry_adapter.get_whitebox_algorithms()

        if not algorithms:
            self._add_system_message(
                "[Warning] No WhiteboxTools algorithms found.\n"
                "Make sure the WhiteboxTools provider is installed and enabled."
            )
            return

        # Group by category
        groups: Dict[str, List[str]] = {}
        for alg in algorithms:
            group = alg.get("group", "Other")
            if group not in groups:
                groups[group] = []
            groups[group].append(f"  • {alg['displayName']}")

        message_parts = [f"Found {len(algorithms)} WhiteboxTools algorithms:\n"]
        for group_name in sorted(groups.keys()):
            message_parts.append(f"\n[{group_name}]")
            message_parts.extend(groups[group_name][:5])  # Show first 5 per group
            if len(groups[group_name]) > 5:
                message_parts.append(f"  ... and {len(groups[group_name]) - 5} more")

        self._add_system_message("\n".join(message_parts))

    def _show_layers(self):
        """Show current QGIS project layers."""
        project = QgsProject.instance()
        layers = project.mapLayers()

        if not layers:
            self._add_system_message(
                "No layers loaded in the current project.\n"
                "Load some raster or vector layers to get started."
            )
            return

        message_parts = [f"Layers in current project ({len(layers)}):\n"]

        for layer_id, layer in layers.items():
            layer_type = "[R]" if layer.type().name == "RasterLayer" else "[V]"
            message_parts.append(f"  {layer_type} {layer.name()}")
            message_parts.append(f"      ID: {layer_id[:40]}...")

        self._add_system_message("\n".join(message_parts))

    def _load_settings(self):
        """Load settings from QSettings."""
        provider = self.settings.value(f"{self.SETTINGS_PREFIX}provider", "ollama")
        index = self.provider_combo.findData(provider)
        if index >= 0:
            self.provider_combo.setCurrentIndex(index)

    def closeEvent(self, event):
        """Handle dock widget close event."""
        # Stop any running worker
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait()

        event.accept()
