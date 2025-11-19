# kloom-litellm

A LiteLLM plugin for intelligent model routing and request tracking with [Kloom](https://kloom.ai).

## Features

- **Intelligent Model Routing** (LiteLLM Proxy): Automatically select the best model based on cost, performance, and availability
- **Request Tracking** (SDK & Proxy): Log all LLM requests and responses for analytics and monitoring
- **Seamless Integration**: Works with any LiteLLM-supported model provider

## Quick Start

### Installation

```bash
pip install kloom-litellm
```

### Basic Usage (Request Tracking)

```python
import litellm
from kloom_litellm import KloomPlugin

# Initialize the plugin
kloom_logger = KloomPlugin()
litellm.callbacks = [kloom_logger]

# Use LiteLLM as normal - requests will automatically be tracked
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Documentation

For complete documentation, including:
- LiteLLM Proxy setup for model routing
- Configuration options
- Advanced usage examples
- Troubleshooting

Visit: **[https://docs.kloom.ai/quickstart/litellm](https://docs.kloom.ai/quickstart/litellm)**

## Development

### From Source

```bash
uv sync
```

## License

See [LICENSE](LICENSE) file.
