#!/usr/bin/env python3
"""
Inspect WhiteboxTools algorithms in QGIS.
Run this script inside QGIS Python console or via:
    qgis_process run script:/path/to/this/script.py

Or run QGIS with: qgis --code inspect_algorithms.py
"""


def inspect_whitebox_algorithms():
    """Inspect WhiteboxTools algorithms and their parameters."""
    from qgis.core import QgsApplication

    registry = QgsApplication.processingRegistry()

    # Find WhiteboxTools provider
    whitebox_provider = None
    for provider in registry.providers():
        pid = provider.id().lower()
        if "wbt" in pid or "whitebox" in pid:
            whitebox_provider = provider
            print(f"Found WhiteboxTools provider: {provider.id()} - {provider.name()}")
            break

    if not whitebox_provider:
        print("WhiteboxTools provider not found!")
        print("Available providers:")
        for p in registry.providers():
            print(f"  - {p.id()}: {p.name()}")
        return

    # Find fill-related algorithms
    print("\n" + "=" * 60)
    print("Fill/Depression related algorithms:")
    print("=" * 60)

    fill_keywords = ["fill", "depression", "sink", "breach"]

    for alg in registry.algorithms():
        if alg.provider() and alg.provider().id() == whitebox_provider.id():
            name_lower = alg.displayName().lower()
            if any(kw in name_lower for kw in fill_keywords):
                print(f"\n{alg.id()}: {alg.displayName()}")
                print(f"  Group: {alg.group()}")

                # Show parameters
                print("  Parameters:")
                for param in alg.parameterDefinitions():
                    flags = param.flags()
                    from qgis.core import QgsProcessingParameterDefinition

                    is_optional = bool(
                        flags & QgsProcessingParameterDefinition.FlagOptional
                    )
                    is_hidden = bool(
                        flags & QgsProcessingParameterDefinition.FlagHidden
                    )

                    if is_hidden:
                        continue

                    req = "optional" if is_optional else "REQUIRED"
                    default = param.defaultValue()
                    print(
                        f"    - {param.name()} ({param.type()}): {param.description()}"
                    )
                    print(f"      [{req}] default={default}")


if __name__ == "__main__":
    inspect_whitebox_algorithms()
