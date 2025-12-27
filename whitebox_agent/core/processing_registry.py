"""
Processing Registry Adapter

Discovers all Processing algorithms at runtime and extracts metadata
for WhiteboxTools algorithms from the QGIS Processing registry.
"""

from typing import Dict, List, Any, Optional
from qgis.core import (
    QgsApplication,
    QgsProcessingRegistry,
    QgsProcessingAlgorithm,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterString,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterBand,
    QgsProcessingParameterCrs,
    QgsProcessingParameterExtent,
    QgsProcessingParameterPoint,
    QgsProcessingParameterFile,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterExpression,
)


class ProcessingRegistryAdapter:
    """
    Adapter for discovering and extracting metadata from QGIS Processing algorithms.

    This class interfaces with the QGIS Processing registry to:
    - Discover all available algorithms
    - Filter algorithms by provider (e.g., WhiteboxTools)
    - Extract parameter and output definitions
    - Provide structured metadata for LLM context
    """

    # Known WhiteboxTools provider IDs (may vary by installation)
    WHITEBOX_PROVIDER_IDS = ["whitebox", "wbt", "whiteboxtools"]

    def __init__(self):
        """Initialize the registry adapter."""
        self._registry: QgsProcessingRegistry = QgsApplication.processingRegistry()
        self._algorithm_cache: Dict[str, Dict[str, Any]] = {}
        self._whitebox_algorithms: List[str] = []

    def get_registry(self) -> QgsProcessingRegistry:
        """Get the QGIS Processing registry instance."""
        return self._registry

    def get_all_providers(self) -> List[Dict[str, str]]:
        """
        Get all available processing providers.

        Returns:
            List of dictionaries with provider id and name.
        """
        providers = []
        for provider in self._registry.providers():
            providers.append(
                {
                    "id": provider.id(),
                    "name": provider.name(),
                    "description": (
                        provider.longName()
                        if hasattr(provider, "longName")
                        else provider.name()
                    ),
                }
            )
        return providers

    def get_whitebox_provider(self) -> Optional[str]:
        """
        Find the WhiteboxTools provider ID.

        Returns:
            The provider ID if found, None otherwise.
        """
        for provider in self._registry.providers():
            provider_id = provider.id().lower()
            for known_id in self.WHITEBOX_PROVIDER_IDS:
                if known_id in provider_id:
                    return provider.id()
        return None

    def get_whitebox_algorithms(
        self, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all WhiteboxTools algorithms.

        Args:
            force_refresh: Force a refresh of the algorithm cache.

        Returns:
            List of algorithm metadata dictionaries.
        """
        if self._whitebox_algorithms and not force_refresh:
            return [
                self.get_algorithm_metadata(alg_id)
                for alg_id in self._whitebox_algorithms
            ]

        self._whitebox_algorithms = []
        algorithms = []

        for alg in self._registry.algorithms():
            provider_id = alg.provider().id().lower() if alg.provider() else ""

            # Check if this is a WhiteboxTools algorithm
            is_whitebox = any(
                wbt_id in provider_id for wbt_id in self.WHITEBOX_PROVIDER_IDS
            )

            if is_whitebox:
                alg_id = alg.id()
                self._whitebox_algorithms.append(alg_id)
                algorithms.append(self.get_algorithm_metadata(alg_id))

        return algorithms

    def get_algorithm_metadata(self, algorithm_id: str) -> Dict[str, Any]:
        """
        Get detailed metadata for a specific algorithm.

        Args:
            algorithm_id: The full algorithm ID (e.g., "whitebox:FillDepressions").

        Returns:
            Dictionary containing algorithm metadata.
        """
        # Check cache first
        if algorithm_id in self._algorithm_cache:
            return self._algorithm_cache[algorithm_id]

        alg = self._registry.algorithmById(algorithm_id)
        if not alg:
            return {"error": f"Algorithm not found: {algorithm_id}"}

        metadata = {
            "id": algorithm_id,
            "name": alg.name(),
            "displayName": alg.displayName(),
            "group": alg.group(),
            "groupId": alg.groupId(),
            "shortDescription": (
                alg.shortDescription() if hasattr(alg, "shortDescription") else ""
            ),
            "shortHelpString": (
                alg.shortHelpString() if hasattr(alg, "shortHelpString") else ""
            ),
            "provider": {
                "id": alg.provider().id() if alg.provider() else "",
                "name": alg.provider().name() if alg.provider() else "",
            },
            "parameters": self._extract_parameters(alg),
            "outputs": self._extract_outputs(alg),
        }

        # Cache the result
        self._algorithm_cache[algorithm_id] = metadata
        return metadata

    def _extract_parameters(self, alg: QgsProcessingAlgorithm) -> List[Dict[str, Any]]:
        """
        Extract parameter definitions from an algorithm.

        Args:
            alg: The processing algorithm.

        Returns:
            List of parameter metadata dictionaries.
        """
        parameters = []

        for param in alg.parameterDefinitions():
            param_info = {
                "name": param.name(),
                "description": param.description(),
                "type": self._get_parameter_type(param),
                "required": not (
                    param.flags() & QgsProcessingParameterDefinition.FlagOptional
                ),
                "advanced": bool(
                    param.flags() & QgsProcessingParameterDefinition.FlagAdvanced
                ),
                "hidden": bool(
                    param.flags() & QgsProcessingParameterDefinition.FlagHidden
                ),
            }

            # Add default value if available
            default = param.defaultValue()
            if default is not None:
                param_info["default"] = (
                    str(default)
                    if not isinstance(default, (bool, int, float))
                    else default
                )

            # Add type-specific metadata
            param_info.update(self._get_parameter_constraints(param))

            parameters.append(param_info)

        return parameters

    def _get_parameter_type(self, param: QgsProcessingParameterDefinition) -> str:
        """
        Get a simplified parameter type string.

        Args:
            param: The parameter definition.

        Returns:
            Human-readable parameter type string.
        """
        type_mapping = {
            QgsProcessingParameterRasterLayer: "raster_layer",
            QgsProcessingParameterVectorLayer: "vector_layer",
            QgsProcessingParameterFeatureSource: "feature_source",
            QgsProcessingParameterRasterDestination: "raster_output",
            QgsProcessingParameterVectorDestination: "vector_output",
            QgsProcessingParameterFileDestination: "file_output",
            QgsProcessingParameterFolderDestination: "folder_output",
            QgsProcessingParameterNumber: "number",
            QgsProcessingParameterBoolean: "boolean",
            QgsProcessingParameterString: "string",
            QgsProcessingParameterEnum: "enum",
            QgsProcessingParameterField: "field",
            QgsProcessingParameterBand: "band",
            QgsProcessingParameterCrs: "crs",
            QgsProcessingParameterExtent: "extent",
            QgsProcessingParameterPoint: "point",
            QgsProcessingParameterFile: "file",
            QgsProcessingParameterMultipleLayers: "multiple_layers",
            QgsProcessingParameterExpression: "expression",
        }

        for param_class, type_name in type_mapping.items():
            if isinstance(param, param_class):
                return type_name

        # Fallback to the parameter's type string
        return param.type()

    def _get_parameter_constraints(
        self, param: QgsProcessingParameterDefinition
    ) -> Dict[str, Any]:
        """
        Get type-specific constraints for a parameter.

        Args:
            param: The parameter definition.

        Returns:
            Dictionary of constraints.
        """
        constraints = {}

        if isinstance(param, QgsProcessingParameterNumber):
            constraints["dataType"] = (
                "integer"
                if param.dataType() == QgsProcessingParameterNumber.Integer
                else "double"
            )
            if param.minimum() is not None:
                constraints["minimum"] = param.minimum()
            if param.maximum() is not None:
                constraints["maximum"] = param.maximum()

        elif isinstance(param, QgsProcessingParameterEnum):
            constraints["options"] = param.options()
            constraints["allowMultiple"] = param.allowMultiple()

        elif isinstance(param, QgsProcessingParameterField):
            constraints["parentLayerParameter"] = param.parentLayerParameterName()
            constraints["dataType"] = param.dataType()

        elif isinstance(param, QgsProcessingParameterBand):
            constraints["parentLayerParameter"] = param.parentLayerParameterName()

        elif isinstance(param, QgsProcessingParameterFile):
            constraints["behavior"] = (
                "folder"
                if param.behavior() == QgsProcessingParameterFile.Folder
                else "file"
            )
            if param.extension():
                constraints["extension"] = param.extension()

        return constraints

    def _extract_outputs(self, alg: QgsProcessingAlgorithm) -> List[Dict[str, Any]]:
        """
        Extract output definitions from an algorithm.

        Args:
            alg: The processing algorithm.

        Returns:
            List of output metadata dictionaries.
        """
        outputs = []

        for output in alg.outputDefinitions():
            output_info = {
                "name": output.name(),
                "description": output.description(),
                "type": output.type(),
            }
            outputs.append(output_info)

        return outputs

    def validate_algorithm_id(self, algorithm_id: str) -> bool:
        """
        Validate that an algorithm ID exists in the registry.

        Args:
            algorithm_id: The algorithm ID to validate.

        Returns:
            True if the algorithm exists, False otherwise.
        """
        return self._registry.algorithmById(algorithm_id) is not None

    def validate_parameters(
        self, algorithm_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate parameters against the algorithm's parameter definitions.

        Note: This validation is lenient - we let QGIS/WhiteboxTools handle
        detailed validation and provide better error messages.

        Args:
            algorithm_id: The algorithm ID.
            params: Dictionary of parameter name to value.

        Returns:
            Dictionary with validation results.
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        alg = self._registry.algorithmById(algorithm_id)
        if not alg:
            result["valid"] = False
            result["errors"].append(f"Algorithm not found: {algorithm_id}")
            return result

        # Just check for unknown parameters as warnings
        param_defs = {p.name(): p for p in alg.parameterDefinitions()}
        for param_name in params:
            if param_name not in param_defs:
                result["warnings"].append(f"Unknown parameter: {param_name}")

        # Don't strictly validate required params - let QGIS handle it
        # This allows algorithms with defaults to work without explicit params
        return result

    def get_algorithm_summary(self, algorithm_id: str) -> str:
        """
        Get a compact summary of an algorithm for LLM context.

        Args:
            algorithm_id: The algorithm ID.

        Returns:
            Formatted string summary.
        """
        metadata = self.get_algorithm_metadata(algorithm_id)

        if "error" in metadata:
            return metadata["error"]

        lines = [
            f"Algorithm: {metadata['displayName']} ({metadata['id']})",
            f"Group: {metadata['group']}",
        ]

        if metadata.get("shortDescription"):
            lines.append(f"Description: {metadata['shortDescription']}")

        # Required parameters
        required_params = [
            p for p in metadata["parameters"] if p["required"] and not p["hidden"]
        ]
        if required_params:
            lines.append("Required parameters:")
            for p in required_params:
                param_line = f"  - {p['name']} ({p['type']}): {p['description']}"
                if "default" in p:
                    param_line += f" [default: {p['default']}]"
                lines.append(param_line)

        # Optional parameters (non-advanced)
        optional_params = [
            p
            for p in metadata["parameters"]
            if not p["required"] and not p["hidden"] and not p["advanced"]
        ]
        if optional_params:
            lines.append("Optional parameters:")
            for p in optional_params:
                lines.append(f"  - {p['name']} ({p['type']}): {p['description']}")

        return "\n".join(lines)

    def search_algorithms(self, query: str) -> List[Dict[str, Any]]:
        """
        Search algorithms by name, description, or group.

        Args:
            query: Search query string.

        Returns:
            List of matching algorithm metadata dictionaries.
        """
        query = query.lower()
        results = []

        for alg in self._registry.algorithms():
            # Check if this is a WhiteboxTools algorithm
            provider_id = alg.provider().id().lower() if alg.provider() else ""
            is_whitebox = any(
                wbt_id in provider_id for wbt_id in self.WHITEBOX_PROVIDER_IDS
            )

            if not is_whitebox:
                continue

            # Search in various fields
            searchable = " ".join(
                [
                    alg.displayName().lower(),
                    alg.name().lower(),
                    alg.group().lower(),
                    (
                        alg.shortDescription()
                        if hasattr(alg, "shortDescription")
                        else ""
                    ).lower(),
                ]
            )

            if query in searchable:
                results.append(self.get_algorithm_metadata(alg.id()))

        return results
