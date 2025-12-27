"""
Settings Dock Widget

Configuration panel for the Whitebox AI Agent.
Allows configuration of LLM providers, API keys, and other settings.
"""

from qgis.PyQt.QtCore import Qt, QSettings
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QComboBox,
    QFormLayout,
    QMessageBox,
    QTabWidget,
    QCheckBox,
    QSpinBox,
)
from qgis.PyQt.QtGui import QFont


class SettingsDockWidget(QDockWidget):
    """Settings panel for configuring the Whitebox AI Agent."""

    SETTINGS_PREFIX = "WhiteboxAgent/"

    # Default models for each provider
    DEFAULT_MODELS = {
        "ollama": "llama3.1",
        "claude": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "gemini": "gemini-2.0-flash",
    }

    # Default base URLs
    DEFAULT_URLS = {
        "ollama": "http://localhost:11434",
        "claude": "https://api.anthropic.com",
        "openai": "https://api.openai.com",
        "gemini": "https://generativelanguage.googleapis.com",
    }

    def __init__(self, iface, parent=None):
        """Initialize the settings dock widget.

        Args:
            iface: QGIS interface instance.
            parent: Parent widget.
        """
        super().__init__("WhiteboxTools AI Settings", parent)
        self.iface = iface
        self.settings = QSettings()

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Set up the settings UI."""
        main_widget = QWidget()
        self.setWidget(main_widget)

        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)

        # Header
        header_label = QLabel("⚙️ Settings")
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # Ollama tab
        ollama_tab = self._create_provider_tab("ollama", requires_api_key=False)
        tab_widget.addTab(ollama_tab, "Ollama")

        # Claude tab
        claude_tab = self._create_provider_tab("claude", requires_api_key=True)
        tab_widget.addTab(claude_tab, "Claude")

        # OpenAI tab
        openai_tab = self._create_provider_tab("openai", requires_api_key=True)
        tab_widget.addTab(openai_tab, "OpenAI")

        # Gemini tab
        gemini_tab = self._create_provider_tab("gemini", requires_api_key=True)
        tab_widget.addTab(gemini_tab, "Gemini")

        # Bedrock tab
        bedrock_tab = self._create_bedrock_tab()
        tab_widget.addTab(bedrock_tab, "Bedrock")

        # Advanced tab
        advanced_tab = self._create_advanced_tab()
        tab_widget.addTab(advanced_tab, "Advanced")

        # Buttons
        button_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self._save_settings)
        self.save_btn.setStyleSheet(
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
        """
        )
        button_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.clicked.connect(self._reset_defaults)
        button_layout.addWidget(self.reset_btn)

        layout.addLayout(button_layout)

        # Status
        self.status_label = QLabel("Settings loaded")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _create_provider_tab(
        self, provider: str, requires_api_key: bool = True
    ) -> QWidget:
        """Create a settings tab for a provider."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Provider settings group
        group = QGroupBox(f"{provider.capitalize()} Configuration")
        form_layout = QFormLayout(group)

        # API Key (if required)
        if requires_api_key:
            api_key_input = QLineEdit()
            api_key_input.setEchoMode(QLineEdit.Password)
            api_key_input.setPlaceholderText(
                f"Enter your {provider.capitalize()} API key..."
            )
            setattr(self, f"{provider}_api_key_input", api_key_input)

            api_key_layout = QHBoxLayout()
            api_key_layout.addWidget(api_key_input)

            show_key_btn = QPushButton("👁")
            show_key_btn.setMaximumWidth(30)
            show_key_btn.setCheckable(True)
            show_key_btn.toggled.connect(
                lambda checked, inp=api_key_input: inp.setEchoMode(
                    QLineEdit.Normal if checked else QLineEdit.Password
                )
            )
            api_key_layout.addWidget(show_key_btn)

            form_layout.addRow("API Key:", api_key_layout)
        else:
            # For Ollama, show info message instead
            info_label = QLabel("No API key required for local Ollama.")
            info_label.setStyleSheet("color: #888; font-style: italic;")
            form_layout.addRow("", info_label)

        # Model name
        model_input = QLineEdit()
        model_input.setPlaceholderText(
            f"Default: {self.DEFAULT_MODELS.get(provider, '')}"
        )
        setattr(self, f"{provider}_model_input", model_input)
        form_layout.addRow("Model:", model_input)

        # Base URL
        url_input = QLineEdit()
        url_input.setPlaceholderText(f"Default: {self.DEFAULT_URLS.get(provider, '')}")
        setattr(self, f"{provider}_url_input", url_input)
        form_layout.addRow("Base URL:", url_input)

        # Test connection button
        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(lambda: self._test_connection(provider))
        form_layout.addRow("", test_btn)

        layout.addWidget(group)

        # Help text
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout(help_group)

        help_texts = {
            "ollama": (
                "Ollama runs locally on your machine.\n\n"
                "1. Install Ollama from https://ollama.ai\n"
                "2. Run: ollama pull llama3.1\n"
                "3. The default URL is http://localhost:11434"
            ),
            "claude": (
                "Claude is Anthropic's AI assistant.\n\n"
                "1. Get an API key from https://console.anthropic.com\n"
                "2. Enter your API key above\n"
                "3. Recommended model: claude-sonnet-4-20250514"
            ),
            "openai": (
                "OpenAI provides GPT models.\n\n"
                "1. Get an API key from https://platform.openai.com\n"
                "2. Enter your API key above\n"
                "3. Recommended model: gpt-4o"
            ),
            "gemini": (
                "Gemini is Google's AI model.\n\n"
                "1. Get an API key from https://aistudio.google.com\n"
                "2. Enter your API key above\n"
                "3. Recommended model: gemini-2.0-flash"
            ),
        }

        help_label = QLabel(help_texts.get(provider, ""))
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #888;")
        help_layout.addWidget(help_label)

        layout.addWidget(help_group)
        layout.addStretch()

        return widget

    def _create_bedrock_tab(self) -> QWidget:
        """Create the Amazon Bedrock settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Bedrock settings group
        group = QGroupBox("Amazon Bedrock Configuration")
        form_layout = QFormLayout(group)

        # Region
        self.bedrock_region_input = QLineEdit()
        self.bedrock_region_input.setPlaceholderText("us-east-1")
        form_layout.addRow("AWS Region:", self.bedrock_region_input)

        # Model ID
        self.bedrock_model_input = QLineEdit()
        self.bedrock_model_input.setPlaceholderText(
            "anthropic.claude-sonnet-4-20250514-v1:0"
        )
        form_layout.addRow("Model ID:", self.bedrock_model_input)

        # Test connection button
        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(lambda: self._test_provider("bedrock"))
        form_layout.addRow("", test_btn)

        layout.addWidget(group)

        # Help text
        help_group = QGroupBox("Setup Instructions")
        help_layout = QVBoxLayout(help_group)

        help_label = QLabel(
            "Amazon Bedrock uses AWS credentials.\n\n"
            "1. Configure AWS credentials:\n"
            "   - Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
            "   - Or use ~/.aws/credentials file\n"
            "2. Ensure you have Bedrock model access enabled\n"
            "3. Install boto3: pip install boto3\n\n"
            "Recommended model: anthropic.claude-sonnet-4-20250514-v1:0"
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #888;")
        help_layout.addWidget(help_label)

        layout.addWidget(help_group)
        layout.addStretch()

        return widget

    def _create_advanced_tab(self) -> QWidget:
        """Create the advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Processing settings
        processing_group = QGroupBox("Processing")
        processing_layout = QFormLayout(processing_group)

        self.auto_load_check = QCheckBox()
        self.auto_load_check.setChecked(True)
        processing_layout.addRow("Auto-load outputs:", self.auto_load_check)

        self.max_context_spin = QSpinBox()
        self.max_context_spin.setRange(10, 500)
        self.max_context_spin.setValue(100)
        processing_layout.addRow("Max algorithms in context:", self.max_context_spin)

        layout.addWidget(processing_group)

        # Debug settings
        debug_group = QGroupBox("Debug")
        debug_layout = QFormLayout(debug_group)

        self.debug_check = QCheckBox()
        self.debug_check.setChecked(False)
        debug_layout.addRow("Debug mode:", self.debug_check)

        self.log_responses_check = QCheckBox()
        self.log_responses_check.setChecked(False)
        debug_layout.addRow("Log LLM responses:", self.log_responses_check)

        layout.addWidget(debug_group)
        layout.addStretch()

        return widget

    def _test_connection(self, provider: str):
        """Test connection to the specified provider."""
        from ..core.llm_client import LLMClient

        # Get current values
        api_key = ""
        if hasattr(self, f"{provider}_api_key_input"):
            api_key = getattr(self, f"{provider}_api_key_input").text()

        model = getattr(self, f"{provider}_model_input").text()
        base_url = getattr(self, f"{provider}_url_input").text()

        # For providers requiring API key
        if provider != "ollama" and not api_key:
            api_key = self.settings.value(
                f"{self.SETTINGS_PREFIX}{provider}_api_key", ""
            )

        if provider != "ollama" and not api_key:
            QMessageBox.warning(
                self,
                "Test Connection",
                f"Please enter an API key for {provider.capitalize()}.",
            )
            return

        try:
            config = LLMClient.create_config(
                provider=provider,
                api_key=api_key or None,
                model=model or None,
                base_url=base_url or None,
            )
            client = LLMClient(config)

            if client.test_connection():
                QMessageBox.information(
                    self,
                    "Test Connection",
                    f"✅ Successfully connected to {provider.capitalize()}!",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Test Connection",
                    f"❌ Failed to connect to {provider.capitalize()}.\n"
                    "Please check your settings.",
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Test Connection",
                f"❌ Connection error:\n{str(e)}",
            )

    def _load_settings(self):
        """Load settings from QSettings."""
        providers = ["ollama", "claude", "openai", "gemini"]

        for provider in providers:
            # API key
            if hasattr(self, f"{provider}_api_key_input"):
                api_key = self.settings.value(
                    f"{self.SETTINGS_PREFIX}{provider}_api_key", ""
                )
                getattr(self, f"{provider}_api_key_input").setText(api_key)

            # Model
            model = self.settings.value(f"{self.SETTINGS_PREFIX}{provider}_model", "")
            getattr(self, f"{provider}_model_input").setText(model)

            # Base URL
            base_url = self.settings.value(
                f"{self.SETTINGS_PREFIX}{provider}_base_url", ""
            )
            getattr(self, f"{provider}_url_input").setText(base_url)

        # Bedrock settings
        self.bedrock_region_input.setText(
            self.settings.value(f"{self.SETTINGS_PREFIX}bedrock_region", "")
        )
        self.bedrock_model_input.setText(
            self.settings.value(f"{self.SETTINGS_PREFIX}bedrock_model", "")
        )

        # Advanced settings
        self.auto_load_check.setChecked(
            self.settings.value(f"{self.SETTINGS_PREFIX}auto_load", True, type=bool)
        )
        self.max_context_spin.setValue(
            self.settings.value(f"{self.SETTINGS_PREFIX}max_context", 100, type=int)
        )
        self.debug_check.setChecked(
            self.settings.value(f"{self.SETTINGS_PREFIX}debug", False, type=bool)
        )
        self.log_responses_check.setChecked(
            self.settings.value(
                f"{self.SETTINGS_PREFIX}log_responses", False, type=bool
            )
        )

        self.status_label.setText("Settings loaded")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")

    def _save_settings(self):
        """Save settings to QSettings."""
        providers = ["ollama", "claude", "openai", "gemini"]

        for provider in providers:
            # API key
            if hasattr(self, f"{provider}_api_key_input"):
                api_key = getattr(self, f"{provider}_api_key_input").text()
                self.settings.setValue(
                    f"{self.SETTINGS_PREFIX}{provider}_api_key", api_key
                )

            # Model
            model = getattr(self, f"{provider}_model_input").text()
            self.settings.setValue(f"{self.SETTINGS_PREFIX}{provider}_model", model)

            # Base URL
            base_url = getattr(self, f"{provider}_url_input").text()
            self.settings.setValue(
                f"{self.SETTINGS_PREFIX}{provider}_base_url", base_url
            )

        # Bedrock settings
        self.settings.setValue(
            f"{self.SETTINGS_PREFIX}bedrock_region", self.bedrock_region_input.text()
        )
        self.settings.setValue(
            f"{self.SETTINGS_PREFIX}bedrock_model", self.bedrock_model_input.text()
        )

        # Advanced settings
        self.settings.setValue(
            f"{self.SETTINGS_PREFIX}auto_load", self.auto_load_check.isChecked()
        )
        self.settings.setValue(
            f"{self.SETTINGS_PREFIX}max_context", self.max_context_spin.value()
        )
        self.settings.setValue(
            f"{self.SETTINGS_PREFIX}debug", self.debug_check.isChecked()
        )
        self.settings.setValue(
            f"{self.SETTINGS_PREFIX}log_responses", self.log_responses_check.isChecked()
        )

        self.settings.sync()

        self.status_label.setText("Settings saved")
        self.status_label.setStyleSheet("color: green; font-size: 10px;")

        self.iface.messageBar().pushSuccess(
            "Whitebox AI Agent",
            "Settings saved successfully!",
        )

    def _reset_defaults(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?\n"
            "This will clear all API keys.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        providers = ["ollama", "claude", "openai", "gemini"]

        for provider in providers:
            if hasattr(self, f"{provider}_api_key_input"):
                getattr(self, f"{provider}_api_key_input").clear()
            getattr(self, f"{provider}_model_input").clear()
            getattr(self, f"{provider}_url_input").clear()

        self.auto_load_check.setChecked(True)
        self.max_context_spin.setValue(100)
        self.debug_check.setChecked(False)
        self.log_responses_check.setChecked(False)

        self.status_label.setText("Defaults restored (not saved)")
        self.status_label.setStyleSheet("color: orange; font-size: 10px;")
