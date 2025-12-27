"""
Agent Executor

Executes agent actions including:
- ask_user: Request information from user
- select_algorithm: Propose algorithm candidates
- run_algorithm: Execute processing algorithms
- explain: Provide explanations
"""

import os
import tempfile
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from qgis.core import (
    QgsApplication,
    QgsProject,
    QgsProcessing,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingAlgorithm,
    QgsRasterLayer,
    QgsVectorLayer,
)

from .processing_registry import ProcessingRegistryAdapter
from .context_builder import ContextBuilder


class ActionType(Enum):
    """Supported agent action types."""

    ASK_USER = "ask_user"
    SELECT_ALGORITHM = "select_algorithm"
    RUN_ALGORITHM = "run_algorithm"
    EXPLAIN = "explain"


@dataclass
class ExecutionResult:
    """Result of an agent action execution."""

    success: bool
    action_type: ActionType
    message: str
    data: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, str]] = None
    error: Optional[str] = None


class AgentFeedback(QgsProcessingFeedback):
    """Custom feedback class to capture processing progress and messages."""

    def __init__(self, progress_callback: Optional[Callable[[float], None]] = None):
        super().__init__()
        self.progress_callback = progress_callback
        self.log_messages: List[str] = []
        self.error_messages: List[str] = []

    def setProgress(self, progress: float):
        """Handle progress updates."""
        super().setProgress(progress)
        if self.progress_callback:
            self.progress_callback(progress)

    def pushInfo(self, info: str):
        """Capture info messages."""
        super().pushInfo(info)
        self.log_messages.append(info)

    def pushWarning(self, warning: str):
        """Capture warning messages."""
        super().pushWarning(warning)
        self.log_messages.append(f"WARNING: {warning}")

    def reportError(self, error: str, fatalError: bool = False):
        """Capture error messages."""
        super().reportError(error, fatalError)
        self.error_messages.append(error)


class AgentExecutor:
    """
    Executes agent actions and manages the processing workflow.

    This class handles:
    - Parsing and validating LLM responses
    - Executing processing algorithms
    - Managing layer loading
    - Providing feedback and results
    """

    def __init__(
        self,
        registry_adapter: ProcessingRegistryAdapter,
        context_builder: ContextBuilder,
    ):
        """
        Initialize the agent executor.

        Args:
            registry_adapter: ProcessingRegistryAdapter instance.
            context_builder: ContextBuilder instance.
        """
        self.registry = registry_adapter
        self.context = context_builder
        self._processing_context: Optional[QgsProcessingContext] = None

    def execute(
        self,
        action: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ExecutionResult:
        """
        Execute an agent action.

        Args:
            action: Parsed action dictionary from LLM.
            progress_callback: Optional callback for progress updates.

        Returns:
            ExecutionResult with the outcome.
        """
        action_type_str = action.get("action", "")

        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            return ExecutionResult(
                success=False,
                action_type=ActionType.EXPLAIN,
                message=f"Unknown action type: {action_type_str}",
                error=f"Valid actions are: {[a.value for a in ActionType]}",
            )

        # Dispatch to appropriate handler
        if action_type == ActionType.ASK_USER:
            return self._handle_ask_user(action)
        elif action_type == ActionType.SELECT_ALGORITHM:
            return self._handle_select_algorithm(action)
        elif action_type == ActionType.RUN_ALGORITHM:
            return self._handle_run_algorithm(action, progress_callback)
        elif action_type == ActionType.EXPLAIN:
            return self._handle_explain(action)
        else:
            return ExecutionResult(
                success=False,
                action_type=action_type,
                message=f"Action handler not implemented: {action_type}",
            )

    def _handle_ask_user(self, action: Dict[str, Any]) -> ExecutionResult:
        """Handle ask_user action."""
        question = action.get("question", "What would you like to do?")

        return ExecutionResult(
            success=True,
            action_type=ActionType.ASK_USER,
            message=question,
            data={"question": question},
        )

    def _handle_select_algorithm(self, action: Dict[str, Any]) -> ExecutionResult:
        """Handle select_algorithm action."""
        candidates = action.get("candidates", [])

        if not candidates:
            return ExecutionResult(
                success=False,
                action_type=ActionType.SELECT_ALGORITHM,
                message="No algorithm candidates provided.",
                error="The 'candidates' field is empty.",
            )

        # Validate each candidate
        valid_candidates = []
        invalid_candidates = []

        for candidate in candidates:
            alg_id = candidate.get("id", "")
            reason = candidate.get("reason", "")

            if self.registry.validate_algorithm_id(alg_id):
                metadata = self.registry.get_algorithm_metadata(alg_id)
                valid_candidates.append(
                    {
                        "id": alg_id,
                        "name": metadata.get("displayName", alg_id),
                        "reason": reason,
                        "group": metadata.get("group", ""),
                    }
                )
            else:
                invalid_candidates.append(alg_id)

        message_parts = []
        if valid_candidates:
            message_parts.append("Suggested algorithms:")
            for i, c in enumerate(valid_candidates, 1):
                message_parts.append(f"  {i}. {c['name']} ({c['id']})")
                message_parts.append(f"     Reason: {c['reason']}")

        if invalid_candidates:
            message_parts.append(
                f"\nWarning: Invalid algorithm IDs: {invalid_candidates}"
            )

        return ExecutionResult(
            success=len(valid_candidates) > 0,
            action_type=ActionType.SELECT_ALGORITHM,
            message="\n".join(message_parts),
            data={
                "candidates": valid_candidates,
                "invalid": invalid_candidates,
            },
        )

    def _handle_run_algorithm(
        self,
        action: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ExecutionResult:
        """Handle run_algorithm action."""
        algorithm_id = action.get("algorithm_id", "")
        params = action.get("params", {})
        load_outputs = action.get("load_outputs", True)

        # Validate algorithm ID
        if not self.registry.validate_algorithm_id(algorithm_id):
            return ExecutionResult(
                success=False,
                action_type=ActionType.RUN_ALGORITHM,
                message=f"Invalid algorithm ID: {algorithm_id}",
                error="Algorithm not found in registry.",
            )

        # Validate parameters
        validation = self.registry.validate_parameters(algorithm_id, params)
        if not validation["valid"]:
            return ExecutionResult(
                success=False,
                action_type=ActionType.RUN_ALGORITHM,
                message=f"Parameter validation failed: {validation['errors']}",
                error=str(validation["errors"]),
            )

        # Resolve layer references in parameters
        resolved_params = self._resolve_params(algorithm_id, params)
        if "error" in resolved_params:
            return ExecutionResult(
                success=False,
                action_type=ActionType.RUN_ALGORITHM,
                message=f"Failed to resolve parameters: {resolved_params['error']}",
                error=resolved_params["error"],
            )

        # Execute the algorithm
        try:
            print(f"[WhiteboxAgent] Running: {algorithm_id}")
            print(f"[WhiteboxAgent] Params: {resolved_params}")

            result = self._run_processing(
                algorithm_id, resolved_params, progress_callback
            )

            print(f"[WhiteboxAgent] Result success: {result['success']}")
            print(f"[WhiteboxAgent] Result outputs: {result.get('outputs', {})}")

            if result["success"]:
                # NOTE: Don't load layers here - must be done on main thread
                # The outputs dict contains the file paths for loading later
                message = f"Successfully executed {algorithm_id}"

                return ExecutionResult(
                    success=True,
                    action_type=ActionType.RUN_ALGORITHM,
                    message=message,
                    data={
                        "algorithm_id": algorithm_id,
                        "load_outputs": load_outputs,  # Flag for caller
                        "log": result.get("log", []),
                    },
                    outputs=result["outputs"],
                )
            else:
                return ExecutionResult(
                    success=False,
                    action_type=ActionType.RUN_ALGORITHM,
                    message=f"Algorithm execution failed: {result.get('error', 'Unknown error')}",
                    error=result.get("error"),
                    data={"log": result.get("log", [])},
                )

        except Exception as e:
            import traceback

            return ExecutionResult(
                success=False,
                action_type=ActionType.RUN_ALGORITHM,
                message=f"Execution error: {str(e)}",
                error=traceback.format_exc(),
            )

    def _handle_explain(self, action: Dict[str, Any]) -> ExecutionResult:
        """Handle explain action."""
        text = action.get("text", "No explanation provided.")

        return ExecutionResult(
            success=True,
            action_type=ActionType.EXPLAIN,
            message=text,
            data={"text": text},
        )

    def _resolve_params(
        self, algorithm_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve layer references and special values in parameters.

        Args:
            algorithm_id: The algorithm ID.
            params: Raw parameter dictionary.

        Returns:
            Resolved parameter dictionary.
        """
        resolved = {}
        metadata = self.registry.get_algorithm_metadata(algorithm_id)
        param_defs = {p["name"]: p for p in metadata.get("parameters", [])}

        for param_name, param_value in params.items():
            param_def = param_defs.get(param_name, {})
            param_type = param_def.get("type", "")

            # Handle TEMP output values - generate actual temp file path
            # WhiteboxTools doesn't handle TEMPORARY_OUTPUT properly
            if param_value == "TEMP" or param_value == "TEMPORARY_OUTPUT":
                import tempfile
                import uuid

                # Generate unique temp file path
                temp_dir = tempfile.gettempdir()
                temp_name = f"wbt_output_{uuid.uuid4().hex[:8]}.tif"
                temp_path = os.path.join(temp_dir, temp_name)
                resolved[param_name] = temp_path
                print(f"[WhiteboxAgent] Generated temp output: {temp_path}")
                continue

            # Convert to string for path checking
            str_value = str(param_value)

            # Check if it's already a valid file path
            if os.path.exists(str_value):
                resolved[param_name] = str_value
                continue

            # Handle layer references for input layers
            if param_type in (
                "raster_layer",
                "vector_layer",
                "feature_source",
                "layer",
                "raster",
                "vector",
            ):
                resolved_ref = self.context.resolve_layer_reference(str_value)
                if resolved_ref:
                    resolved[param_name] = resolved_ref
                else:
                    # Try to find by partial match on layer name
                    project = QgsProject.instance()
                    for layer in project.mapLayers().values():
                        if str_value.lower() in layer.name().lower():
                            resolved[param_name] = layer.source()
                            break
                    else:
                        return {
                            "error": f"Could not resolve layer reference: {param_value}"
                        }
            else:
                resolved[param_name] = param_value

        return resolved

    def _run_processing(
        self,
        algorithm_id: str,
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run a processing algorithm.

        Args:
            algorithm_id: The algorithm ID.
            params: Resolved parameter dictionary.
            progress_callback: Optional progress callback.

        Returns:
            Dictionary with success status, outputs, and log.
        """
        import processing

        # Create feedback
        feedback = AgentFeedback(progress_callback)

        # Create context
        context = QgsProcessingContext()
        context.setProject(QgsProject.instance())

        try:
            # Run the algorithm
            result = processing.run(
                algorithm_id,
                params,
                context=context,
                feedback=feedback,
            )

            return {
                "success": True,
                "outputs": result,
                "log": feedback.log_messages,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "log": feedback.log_messages + feedback.error_messages,
            }

    def _load_outputs(self, outputs: Dict[str, Any]) -> List[str]:
        """
        Load output layers into the QGIS project.

        Args:
            outputs: Dictionary of output names to paths/values.

        Returns:
            List of loaded layer names.
        """
        loaded = []
        project = QgsProject.instance()

        # Debug: print outputs structure
        print(f"[WhiteboxAgent] Loading outputs: {outputs}")

        for output_name, output_value in outputs.items():
            print(f"[WhiteboxAgent] Checking output: {output_name} = {output_value}")

            if output_value is None:
                print(f"[WhiteboxAgent] Skipping {output_name}: None value")
                continue

            # Skip non-path outputs
            if not isinstance(output_value, str):
                print(
                    f"[WhiteboxAgent] Skipping {output_name}: not a string ({type(output_value)})"
                )
                continue

            # Check if it's a file path
            if not os.path.exists(output_value):
                print(
                    f"[WhiteboxAgent] Skipping {output_name}: file does not exist at {output_value}"
                )
                continue

            # Determine layer type and load
            layer_name = os.path.splitext(os.path.basename(output_value))[0]
            print(f"[WhiteboxAgent] Attempting to load: {output_value} as {layer_name}")

            # Try as raster first (common for WhiteboxTools)
            layer = QgsRasterLayer(output_value, layer_name)
            if layer.isValid():
                project.addMapLayer(layer)
                loaded.append(layer_name)
                print(f"[WhiteboxAgent] Successfully loaded raster: {layer_name}")
                continue

            # Try as vector
            layer = QgsVectorLayer(output_value, layer_name, "ogr")
            if layer.isValid():
                project.addMapLayer(layer)
                loaded.append(layer_name)
                print(f"[WhiteboxAgent] Successfully loaded vector: {layer_name}")
                continue

            print(f"[WhiteboxAgent] Failed to load {output_value} as raster or vector")

        print(f"[WhiteboxAgent] Total loaded layers: {loaded}")
        return loaded

    def get_algorithm_help(self, algorithm_id: str) -> str:
        """
        Get detailed help for an algorithm.

        Args:
            algorithm_id: The algorithm ID.

        Returns:
            Formatted help text.
        """
        return self.registry.get_algorithm_summary(algorithm_id)

    def list_available_algorithms(self) -> List[Dict[str, str]]:
        """
        List all available WhiteboxTools algorithms.

        Returns:
            List of algorithm info dictionaries.
        """
        algorithms = self.registry.get_whitebox_algorithms()
        return [
            {
                "id": alg["id"],
                "name": alg["displayName"],
                "group": alg["group"],
            }
            for alg in algorithms
        ]
