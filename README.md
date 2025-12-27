# Whitebox AI Agent

An AI-powered QGIS plugin for running WhiteboxTools through natural language using Ollama, Claude, OpenAI, or Gemini as the reasoning engine.

![QGIS Version](https://img.shields.io/badge/QGIS-%E2%89%A53.28-green)
![License](https://img.shields.io/badge/License-MIT-blue)

## Features

- **Natural Language Interface**: Describe your analysis in plain English
- **Multiple LLM Backends**: Ollama (local), Claude, OpenAI, Gemini
- **Dynamic Algorithm Discovery**: Automatically discovers WhiteboxTools algorithms from QGIS Processing
- **Smart Parameter Validation**: Validates all parameters against algorithm schemas
- **Automatic Output Loading**: Results are automatically added to your QGIS project
- **Safe Execution**: Uses QGIS Processing API, not shell commands

## Installation

### Prerequisites

1. **QGIS 3.28+** installed
2. **WhiteboxTools Processing Provider** installed and enabled in QGIS
3. **Ollama** (for local LLM) or API keys for Claude/OpenAI/Gemini

### Install Plugin

```bash
# Clone the repository
git clone https://github.com/opengeos/qgis-whitebox-agent.git
cd whitebox-agent

# Install to QGIS plugins directory
python install_plugin.py
```

Or manually copy the `whitebox_agent` folder to your QGIS plugins directory:
- **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
- **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
- **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`

### Enable Plugin

1. Restart QGIS
2. Go to **Plugins → Manage and Install Plugins**
3. Search for "Whitebox AI Agent"
4. Check the box to enable it

## Configuration

### Using Ollama (Recommended for Local Use)

1. Install Ollama from https://ollama.ai
2. Pull a model: `ollama pull llama3.1`
3. Start Ollama: `ollama serve`
4. In the plugin settings, select "Ollama" as the provider

### Using Claude

1. Get an API key from https://console.anthropic.com
2. In the plugin Settings panel, enter your API key
3. Select "Claude" as the provider

### Using OpenAI

1. Get an API key from https://platform.openai.com
2. In the plugin Settings panel, enter your API key
3. Select "OpenAI" as the provider

### Using Gemini

1. Get an API key from https://aistudio.google.com
2. In the plugin Settings panel, enter your API key
3. Select "Gemini" as the provider

## Usage

### Basic Workflow

1. **Load Data**: Add raster/vector layers to your QGIS project
2. **Open Chat Panel**: Click the Whitebox AI Agent icon in the toolbar
3. **Ask Questions**: Type natural language requests like:
   - "Fill depressions in my DEM"
   - "Calculate flow accumulation"
   - "Extract stream network with threshold 1000"
   - "What algorithms can I use for watershed analysis?"

### Example Workflow: Hydrological Analysis

```
User: Fill the sinks in my DEM layer

Agent: ✅ Successfully executed whitebox:FillDepressions
       Loaded layers: dem_filled

User: Now calculate flow accumulation

Agent: ✅ Successfully executed whitebox:D8FlowAccumulation
       Loaded layers: flow_accumulation

User: Extract streams with a threshold of 500

Agent: ✅ Successfully executed whitebox:ExtractStreams
       Loaded layers: streams
```

## Architecture

```
whitebox_agent/
├── __init__.py                 # Plugin entry point
├── whitebox_agent.py           # Main plugin class
├── metadata.txt                # Plugin metadata
├── core/
│   ├── processing_registry.py  # Algorithm discovery
│   ├── context_builder.py      # LLM context generation
│   ├── llm_client.py           # Multi-provider LLM client
│   └── agent_executor.py       # Action execution
├── dialogs/
│   ├── chat_dock.py            # Chat interface
│   └── settings_dock.py        # Settings panel
└── icons/
    ├── icon.svg
    ├── settings.svg
    └── about.svg
```

### Components

1. **Processing Registry Adapter**: Discovers all WhiteboxTools algorithms from QGIS Processing registry at runtime
2. **Context Builder**: Builds structured context including available algorithms and loaded layers
3. **LLM Client**: Unified client supporting Ollama, Claude, OpenAI, and Gemini
4. **Agent Executor**: Executes actions returned by the LLM (run algorithm, ask user, explain)

## LLM Response Format

The LLM returns structured JSON responses:

```json
{
  "action": "run_algorithm",
  "algorithm_id": "whitebox:FillDepressions",
  "params": {
    "dem": "layer_id_here",
    "output": "TEMP"
  },
  "load_outputs": true
}
```

Supported actions:
- `ask_user`: Request information from the user
- `select_algorithm`: Suggest candidate algorithms
- `run_algorithm`: Execute a processing algorithm
- `explain`: Provide explanation or information

## Testing

```bash
# Test outside QGIS (LLM client only)
python test_plugin.py
```

For full testing, the plugin must be run inside QGIS with WhiteboxTools installed.

## Sample Data

The `data/` folder contains sample datasets for testing:
- `dem.tif`: Sample Digital Elevation Model
- `basin.geojson`: Sample watershed boundary

## Development

### Requirements

- Python 3.10+
- PyQt5 (bundled with QGIS)
- QGIS 3.28+

### Testing with geo conda environment

```bash
# Activate the geo conda environment
conda activate geo

# Run QGIS from command line
qgis
```

## Safety Features

- **Algorithm Validation**: All algorithm IDs are validated against the QGIS Processing registry
- **Parameter Validation**: Parameters are validated against algorithm schemas
- **Layer Validation**: Input layers are validated against the current project
- **No Shell Execution**: All processing runs through QGIS Processing API
- **Temporary Outputs**: Uses QGIS temporary outputs by default

## Troubleshooting

### No algorithms found

Make sure WhiteboxTools Processing provider is installed:
1. Go to **Plugins → Manage and Install Plugins**
2. Search for "WhiteboxTools" or "WBT"
3. Install and enable the WhiteboxTools plugin

### Connection errors with Ollama

1. Make sure Ollama is running: `ollama serve`
2. Check if the model is pulled: `ollama list`
3. Verify the base URL in settings (default: `http://localhost:11434`)

### API key errors

1. Open the Settings panel
2. Enter your API key for the selected provider
3. Click "Test Connection" to verify
4. Save settings

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgements

- [WhiteboxTools](https://github.com/jblindsay/whitebox-tools) by John Lindsay
- [QGIS](https://qgis.org) - The leading Free and Open Source GIS
- [Ollama](https://ollama.ai) - Run LLMs locally
