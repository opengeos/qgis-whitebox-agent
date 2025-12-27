#!/usr/bin/env python3
"""
Whitebox AI Agent - Test Script

This script tests the plugin components outside of QGIS.
It can be used to verify the LLM client and other non-QGIS components.

Usage:
    python test_plugin.py

Note: Full testing requires running inside QGIS with the geo conda environment.
"""

import sys
import json


def test_llm_client():
    """Test the LLM client with Ollama."""
    print("\n=== Testing LLM Client ===\n")

    try:
        from whitebox_agent.core.llm_client import LLMClient, LLMConfig, LLMProvider

        # Test with Ollama (local, no API key needed)
        print("Testing Ollama connection...")

        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.1",
        )

        client = LLMClient(config)

        # Simple test
        test_context = """
=== LAYER_CATALOG ===
RASTER LAYERS:
  - layer_id: dem_layer_123
    name: dem
    type: raster
    crs: EPSG:4326
    dimensions: 1000x1000
    bands: 1

=== AVAILABLE_ALGORITHMS ===
[Hydrological Analysis]
  - whitebox:FillDepressions: Fill Depressions
      Fills all depressions in a DEM using Planchon and Darboux algorithm
  - whitebox:D8FlowAccumulation: D8 Flow Accumulation
      Calculates flow accumulation using D8 algorithm
"""

        print("Sending test message...")
        response = client.send_message(
            user_message="Fill the depressions in my DEM layer",
            context=test_context,
        )

        print("\nResponse:")
        print(json.dumps(response, indent=2))

        if "action" in response:
            print("\n✅ LLM Client test passed!")
            return True
        else:
            print("\n⚠️ Response doesn't contain expected 'action' field")
            return False

    except ImportError as e:
        print(f"Import error (expected outside QGIS): {e}")
        return None
    except ConnectionError as e:
        print(f"Connection error: {e}")
        print("\nMake sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_json_parsing():
    """Test JSON response parsing."""
    print("\n=== Testing JSON Parsing ===\n")

    try:
        from whitebox_agent.core.llm_client import LLMClient, LLMConfig, LLMProvider

        config = LLMConfig(provider=LLMProvider.OLLAMA)
        client = LLMClient(config)

        # Test various response formats
        test_cases = [
            # Clean JSON
            ('{"action": "explain", "text": "test"}', True),
            # JSON in markdown
            ('```json\n{"action": "run_algorithm", "algorithm_id": "test"}\n```', True),
            # With extra text (should still extract JSON)
            (
                'Here is my response:\n{"action": "ask_user", "question": "Which layer?"}\n',
                True,
            ),
            # Invalid JSON
            ("This is not JSON", False),
        ]

        all_passed = True
        for text, should_succeed in test_cases:
            result = client._parse_response(text)
            has_action = "action" in result and not result.get("error")
            passed = has_action == should_succeed

            status = "✅" if passed else "❌"
            print(f"{status} Parse test: {text[:50]}...")

            if not passed:
                all_passed = False
                print(
                    f"   Expected success={should_succeed}, got action={result.get('action')}"
                )

        return all_passed

    except ImportError as e:
        print(f"Import error (expected outside QGIS): {e}")
        return None


def main():
    """Run all tests."""
    print("=" * 60)
    print("Whitebox AI Agent - Test Suite")
    print("=" * 60)

    results = {}

    # Add the plugin to path
    import os

    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    # Run tests
    results["json_parsing"] = test_json_parsing()
    results["llm_client"] = test_llm_client()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⏭️ SKIPPED"
        print(f"  {test_name}: {status}")

    print()

    # Return exit code
    failed = sum(1 for r in results.values() if r is False)
    return failed


if __name__ == "__main__":
    sys.exit(main())
