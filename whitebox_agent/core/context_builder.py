"""
Context Builder

Builds structured text blocks for LLM input including:
- Available algorithms catalog
- Layer catalog from QGIS project
- Parameter schemas for specific algorithms
"""

from typing import Dict, List, Any, Optional
from qgis.core import (
    QgsProject,
    QgsMapLayer,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsWkbTypes,
)

from .processing_registry import ProcessingRegistryAdapter


class ContextBuilder:
    """
    Builds structured context blocks for LLM prompts.

    This class creates formatted text representations of:
    - Available WhiteboxTools algorithms
    - Current QGIS project layers
    - Algorithm parameter schemas
    """

    def __init__(self, registry_adapter: ProcessingRegistryAdapter):
        """
        Initialize the context builder.

        Args:
            registry_adapter: ProcessingRegistryAdapter instance.
        """
        self.registry = registry_adapter

    def build_full_context(
        self,
        include_algorithms: bool = True,
        include_layers: bool = True,
        algorithm_id: Optional[str] = None,
    ) -> str:
        """
        Build the complete context for an LLM prompt.

        Args:
            include_algorithms: Include available algorithms catalog.
            include_layers: Include layer catalog.
            algorithm_id: If provided, include detailed schema for this algorithm.

        Returns:
            Formatted context string.
        """
        sections = []

        if include_layers:
            sections.append(self.build_layer_catalog())

        if include_algorithms:
            sections.append(self.build_algorithm_catalog())

        if algorithm_id:
            sections.append(self.build_param_schema(algorithm_id))

        return "\n\n".join(sections)

    def build_layer_catalog(self) -> str:
        """
        Build the LAYER_CATALOG context block.

        Returns:
            Formatted string listing all layers in the current QGIS project.
        """
        lines = ["=== LAYER_CATALOG ==="]
        lines.append(
            "IMPORTANT: Use the 'source:' path below as algorithm input parameters!"
        )
        lines.append("")

        project = QgsProject.instance()
        layers = project.mapLayers()

        if not layers:
            lines.append(
                "(No layers currently loaded - ask user to load a layer first)"
            )
            return "\n".join(lines)

        raster_layers = []
        vector_layers = []
        other_layers = []

        for layer_id, layer in layers.items():
            layer_info = self._get_layer_info(layer)

            if isinstance(layer, QgsRasterLayer):
                raster_layers.append(layer_info)
            elif isinstance(layer, QgsVectorLayer):
                vector_layers.append(layer_info)
            else:
                other_layers.append(layer_info)

        if raster_layers:
            lines.append("RASTER LAYERS:")
            for info in raster_layers:
                lines.append(self._format_layer_info(info))
            lines.append("")

        if vector_layers:
            lines.append("VECTOR LAYERS:")
            for info in vector_layers:
                lines.append(self._format_layer_info(info))
            lines.append("")

        if other_layers:
            lines.append("OTHER LAYERS:")
            for info in other_layers:
                lines.append(self._format_layer_info(info))
            lines.append("")

        return "\n".join(lines)

    def _get_layer_info(self, layer: QgsMapLayer) -> Dict[str, Any]:
        """
        Extract information about a layer.

        Args:
            layer: The QGIS map layer.

        Returns:
            Dictionary of layer information.
        """
        info = {
            "id": layer.id(),
            "name": layer.name(),
            "type": "unknown",
            "source": layer.source(),
            "crs": layer.crs().authid() if layer.crs().isValid() else "unknown",
        }

        if isinstance(layer, QgsRasterLayer):
            info["type"] = "raster"
            info["width"] = layer.width()
            info["height"] = layer.height()
            info["bandCount"] = layer.bandCount()
            info["dataType"] = (
                layer.dataProvider().dataType(1).name
                if layer.dataProvider()
                else "unknown"
            )

        elif isinstance(layer, QgsVectorLayer):
            info["type"] = "vector"
            info["geometryType"] = QgsWkbTypes.displayString(layer.wkbType())
            info["featureCount"] = layer.featureCount()
            info["fields"] = [f.name() for f in layer.fields()]

        return info

    def _format_layer_info(self, info: Dict[str, Any]) -> str:
        """
        Format layer information as a string.

        Args:
            info: Layer information dictionary.

        Returns:
            Formatted string.
        """
        lines = [f"  - name: {info['name']}"]
        lines.append(f"    source: {info['source']}")  # Important for algorithm params
        lines.append(f"    layer_id: {info['id']}")
        lines.append(f"    type: {info['type']}")
        lines.append(f"    crs: {info['crs']}")

        if info["type"] == "raster":
            lines.append(f"    dimensions: {info['width']}x{info['height']}")
            lines.append(f"    bands: {info['bandCount']}")

        elif info["type"] == "vector":
            lines.append(f"    geometry: {info['geometryType']}")
            lines.append(f"    features: {info['featureCount']}")
            if info.get("fields"):
                lines.append(f"    fields: {', '.join(info['fields'][:10])}")
                if len(info["fields"]) > 10:
                    lines.append(f"            ... and {len(info['fields']) - 10} more")

        return "\n".join(lines)

    def build_algorithm_catalog(self, max_algorithms: int = 100) -> str:
        """
        Build the AVAILABLE_ALGORITHMS context block.

        Args:
            max_algorithms: Maximum number of algorithms to include.

        Returns:
            Formatted string listing available WhiteboxTools algorithms.
        """
        lines = ["=== AVAILABLE_ALGORITHMS ==="]
        lines.append("WhiteboxTools algorithms available through QGIS Processing.")
        lines.append("IMPORTANT: Use the EXACT algorithm_id (e.g., 'wbt:FillDepressions') when running!")
        lines.append("")

        algorithms = self.registry.get_whitebox_algorithms()

        if not algorithms:
            lines.append(
                "(No WhiteboxTools algorithms found. Ensure WhiteboxTools provider is installed.)"
            )
            return "\n".join(lines)

        # Common hydrology algorithms - show with parameters
        common_keywords = [
            "fill",
            "depression",
            "sink",
            "flow",
            "accumulation",
            "stream",
            "watershed",
            "basin",
            "breach",
            "d8",
            "slope",
            "aspect",
            "hillshade",
            "wetness",
            "twi",
            "tpi",
            "index",
            "curvature",
            "relief",
        ]

        # Group by algorithm group
        groups: Dict[str, List[Dict]] = {}
        for alg in algorithms[:max_algorithms]:
            group = alg.get("group", "Other")
            if group not in groups:
                groups[group] = []
            groups[group].append(alg)

        for group_name in sorted(groups.keys()):
            lines.append(f"[{group_name}]")
            for alg in groups[group_name]:
                alg_name = alg.get("displayName", "").lower()
                alg_id = alg["id"]

                # Check if this is a common algorithm
                is_common = any(kw in alg_name for kw in common_keywords)

                lines.append(f"  - {alg_id}: {alg['displayName']}")

                # For common algorithms, show required parameters
                if is_common:
                    required_params = [
                        p
                        for p in alg.get("parameters", [])
                        if p.get("required") and not p.get("hidden")
                    ]
                    if required_params:
                        param_strs = []
                        for p in required_params[:4]:  # Max 4 params
                            param_strs.append(f"{p['name']}({p['type']})")
                        lines.append(f"      params: {', '.join(param_strs)}")
            lines.append("")

        if len(algorithms) > max_algorithms:
            lines.append(f"... and {len(algorithms) - max_algorithms} more algorithms")

        return "\n".join(lines)

    def build_param_schema(self, algorithm_id: str) -> str:
        """
        Build the PARAM_SCHEMA context block for a specific algorithm.

        Args:
            algorithm_id: The algorithm ID.

        Returns:
            Formatted string with detailed parameter schema.
        """
        lines = ["=== PARAM_SCHEMA ==="]
        lines.append(f"Parameter schema for: {algorithm_id}")
        lines.append("")

        metadata = self.registry.get_algorithm_metadata(algorithm_id)

        if "error" in metadata:
            lines.append(f"ERROR: {metadata['error']}")
            return "\n".join(lines)

        lines.append(f"Name: {metadata['displayName']}")
        lines.append(f"Group: {metadata['group']}")

        if metadata.get("shortDescription"):
            lines.append(f"Description: {metadata['shortDescription']}")

        if metadata.get("shortHelpString"):
            help_text = metadata["shortHelpString"][:500]
            lines.append(f"Help: {help_text}")

        lines.append("")

        # Parameters section
        lines.append("PARAMETERS:")
        params = metadata.get("parameters", [])

        required_params = [p for p in params if p["required"] and not p["hidden"]]
        optional_params = [p for p in params if not p["required"] and not p["hidden"]]

        if required_params:
            lines.append("  Required:")
            for p in required_params:
                lines.append(self._format_param_schema(p, indent=4))

        if optional_params:
            lines.append("  Optional:")
            for p in optional_params:
                lines.append(self._format_param_schema(p, indent=4))

        # Outputs section
        lines.append("")
        lines.append("OUTPUTS:")
        outputs = metadata.get("outputs", [])

        for out in outputs:
            lines.append(f"  - {out['name']} ({out['type']}): {out['description']}")

        return "\n".join(lines)

    def _format_param_schema(self, param: Dict[str, Any], indent: int = 2) -> str:
        """
        Format a parameter schema entry.

        Args:
            param: Parameter metadata dictionary.
            indent: Indentation spaces.

        Returns:
            Formatted string.
        """
        prefix = " " * indent
        lines = [f"{prefix}- {param['name']} ({param['type']}):"]
        lines.append(f"{prefix}    description: {param['description']}")

        if "default" in param:
            lines.append(f"{prefix}    default: {param['default']}")

        if "options" in param:
            lines.append(f"{prefix}    options: {param['options']}")

        if "minimum" in param or "maximum" in param:
            range_str = f"{param.get('minimum', '-∞')} to {param.get('maximum', '∞')}"
            lines.append(f"{prefix}    range: {range_str}")

        if param.get("dataType"):
            lines.append(f"{prefix}    dataType: {param['dataType']}")

        return "\n".join(lines)

    def build_compact_algorithm_list(self) -> str:
        """
        Build a compact list of algorithm IDs and names for quick reference.

        Returns:
            Formatted string with algorithm list.
        """
        algorithms = self.registry.get_whitebox_algorithms()

        lines = ["Available WhiteboxTools algorithms:"]
        for alg in algorithms:
            lines.append(f"- {alg['id']}: {alg['displayName']}")

        return "\n".join(lines)

    def get_layer_by_name(self, name: str) -> Optional[QgsMapLayer]:
        """
        Find a layer by name (case-insensitive).

        Args:
            name: Layer name to search for.

        Returns:
            The layer if found, None otherwise.
        """
        project = QgsProject.instance()

        # Try exact match first
        layers = project.mapLayersByName(name)
        if layers:
            return layers[0]

        # Try case-insensitive match
        name_lower = name.lower()
        for layer in project.mapLayers().values():
            if layer.name().lower() == name_lower:
                return layer

        return None

    def get_layer_by_id(self, layer_id: str) -> Optional[QgsMapLayer]:
        """
        Find a layer by ID.

        Args:
            layer_id: Layer ID to search for.

        Returns:
            The layer if found, None otherwise.
        """
        project = QgsProject.instance()
        return project.mapLayer(layer_id)

    def resolve_layer_reference(self, reference: str) -> Optional[str]:
        """
        Resolve a layer reference to a layer ID or path.

        The reference can be:
        - A layer ID
        - A layer name
        - A file path

        Args:
            reference: The layer reference string.

        Returns:
            Resolved layer ID or path, None if not found.
        """
        import os

        # Check if it's a file path
        if os.path.exists(reference):
            return reference

        # Check if it's a layer ID
        layer = self.get_layer_by_id(reference)
        if layer:
            return layer.source()

        # Check if it's a layer name
        layer = self.get_layer_by_name(reference)
        if layer:
            return layer.source()

        return None
