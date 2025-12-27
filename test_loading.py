#!/usr/bin/env python3
"""
Test script to debug layer loading in QGIS.
Run this in QGIS Python Console.
"""

import processing
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProject,
    QgsRasterLayer,
)

# Test FillDepressions
dem_path = "/media/hdd/Dropbox/GitHub/whitebox-agents/data/dem.tif"

print("=" * 60)
print("Testing wbt:FillDepressions")
print("=" * 60)

# Create context and feedback
context = QgsProcessingContext()
context.setProject(QgsProject.instance())
feedback = QgsProcessingFeedback()

# Run the algorithm
params = {"dem": dem_path, "output": "TEMPORARY_OUTPUT"}  # Standard QGIS way

print(f"Input params: {params}")

try:
    result = processing.run(
        "wbt:FillDepressions", params, context=context, feedback=feedback
    )
    print(f"\nResult: {result}")
    print(f"Result type: {type(result)}")

    # Check each output
    for key, value in result.items():
        print(f"\n  {key}: {value}")
        print(f"    Type: {type(value)}")

        if isinstance(value, str):
            import os

            print(f"    Exists: {os.path.exists(value)}")

            if os.path.exists(value):
                # Try to load
                layer = QgsRasterLayer(value, "test_filled_dem")
                print(f"    Valid layer: {layer.isValid()}")

                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
                    print(f"    Added to project!")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
