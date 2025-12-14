"""
PSA Logging System - Records configuration and sparsity metrics
Adapted for PaddlePaddle
"""
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class PSALogger:
    """
    Comprehensive logger for Pyramid Sparse Attention.
    Tracks configuration, per-layer sparsity, and provides summary statistics.
    """

    def __init__(
        self,
        log_dir: str,
        config: Any,  # AttentionConfig
        layer_idx: int = -1,
        session_name: Optional[str] = None,
    ):
        """
        Initialize PSA Logger.

        Args:
            log_dir: Directory to save logs
            config: AttentionConfig instance
            layer_idx: Layer index (-1 for shared module)
            session_name: Optional session identifier
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.layer_idx = layer_idx

        # Create session timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"psa_session_{timestamp}"

        # Initialize log files
        self.config_file = self.log_dir / f"{self.session_name}_config.json"
        self.sparsity_file = self.log_dir / f"{self.session_name}_sparsity.jsonl"
        self.summary_file = self.log_dir / f"{self.session_name}_summary.txt"

        # Open sparsity log file handle
        self.sparsity_handle = open(self.sparsity_file, "a", encoding="utf-8")

        # Tracking variables
        self.layer_stats: Dict[int, Dict[str, Any]] = {}
        self.global_counter = 0

        # Write initial configuration
        self._write_config()
        self._write_startup_banner()

    def _write_config(self):
        """Write configuration to JSON file (silent)."""
        config_dict = {
            "session_name": self.session_name,
            "timestamp": datetime.now().isoformat(),
            "layer_idx": self.layer_idx,
            "config": asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else str(self.config),
        }

        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

    def _write_startup_banner(self):
        """Write startup info to file only."""
        banner_lines = [
            f"PSA Logger - {self.session_name}",
            f"Layer: {self.layer_idx}, Mode: {self.config.mask_mode}, Block: {self.config.query_block}",
        ]
        banner = "\n".join(banner_lines)

        # Write to summary file only (no console output)
        with open(self.summary_file, "w", encoding="utf-8") as f:
            f.write(banner + "\n")

    def log_sparsity(
        self,
        layer_idx: int,
        sparsity: float,
        per_head_density: List[float],
        sequence_length: int,
        batch_size: int = 1,
        num_heads: int = -1,
        qkv_shape: Optional[Dict[str, tuple]] = None,
    ):
        """
        Log sparsity information for a single forward pass.

        Args:
            layer_idx: Current layer index
            sparsity: Overall sparsity value (0-1, higher = more sparse)
            per_head_density: Density per attention head
            sequence_length: Input sequence length
            batch_size: Batch size
            num_heads: Number of attention heads
            qkv_shape: Dict with Q, K, V tensor shapes
        """
        self.global_counter += 1

        # Initialize layer stats if needed
        if layer_idx not in self.layer_stats:
            self.layer_stats[layer_idx] = {
                "count": 0,
                "total_sparsity": 0.0,
                "min_sparsity": float("inf"),
                "max_sparsity": 0.0,
                "sparsity_history": [],
                "qkv_shapes": [],
            }

        # Update layer statistics
        stats = self.layer_stats[layer_idx]
        stats["count"] += 1
        stats["total_sparsity"] += sparsity
        stats["min_sparsity"] = min(stats["min_sparsity"], sparsity)
        stats["max_sparsity"] = max(stats["max_sparsity"], sparsity)
        stats["sparsity_history"].append(sparsity)
        if qkv_shape:
            stats["qkv_shapes"].append(qkv_shape)

        # Keep only last 100 entries in history
        if len(stats["sparsity_history"]) > 100:
            stats["sparsity_history"] = stats["sparsity_history"][-100:]
        if len(stats.get("qkv_shapes", [])) > 100:
            stats["qkv_shapes"] = stats["qkv_shapes"][-100:]

        # Write to JSONL file
        entry = {
            "global_step": self.global_counter,
            "timestamp": datetime.now().isoformat(),
            "layer_idx": layer_idx,
            "sparsity": round(sparsity, 6),
            "density": round(1 - sparsity, 6),
            "per_head_density": [round(d, 6) for d in per_head_density],
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "num_heads": num_heads if num_heads > 0 else len(per_head_density),
        }
        if qkv_shape:
            entry["qkv_shape"] = {k: list(v) for k, v in qkv_shape.items()}

        self.sparsity_handle.write(json.dumps(entry) + "\n")
        self.sparsity_handle.flush()

    def print_progress(self, layer_idx: int, interval: int = 200):
        """Print progress at intervals (minimal output)."""
        pass  # Disabled for cleaner output

    def write_summary(self):
        """Write comprehensive summary statistics."""
        if not self.layer_stats:
            return

        summary_lines = [
            "",
            "=" * 80,
            f"PSA Sparsity Summary - {self.session_name}",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Forward Passes: {self.global_counter}",
            "",
            "Per-Layer Statistics:",
            "-" * 80,
        ]

        # Table header
        summary_lines.append(
            f"{'Layer':<8} {'Count':<8} {'Avg Sparsity':<15} {'Min':<10} {'Max':<10} {'StdDev':<10}"
        )
        summary_lines.append("-" * 80)

        # Per-layer statistics
        for layer_idx in sorted(self.layer_stats.keys()):
            stats = self.layer_stats[layer_idx]
            avg_sparsity = stats["total_sparsity"] / stats["count"]

            # Calculate standard deviation
            history = stats["sparsity_history"]
            if len(history) > 1:
                mean = sum(history) / len(history)
                variance = sum((x - mean) ** 2 for x in history) / len(history)
                std_dev = variance ** 0.5
            else:
                std_dev = 0.0

            summary_lines.append(
                f"{layer_idx:<8} {stats['count']:<8} "
                f"{avg_sparsity:<15.6f} "
                f"{stats['min_sparsity']:<10.6f} "
                f"{stats['max_sparsity']:<10.6f} "
                f"{std_dev:<10.6f}"
            )

        summary_lines.extend([
            "-" * 80,
            "",
            "Overall Statistics:",
            "-" * 80,
        ])

        # Overall statistics
        all_sparsities = []
        total_count = 0
        total_sparsity = 0.0

        for stats in self.layer_stats.values():
            all_sparsities.extend(stats["sparsity_history"])
            total_count += stats["count"]
            total_sparsity += stats["total_sparsity"]

        if all_sparsities:
            overall_avg = total_sparsity / total_count
            overall_min = min(all_sparsities)
            overall_max = max(all_sparsities)

            mean = sum(all_sparsities) / len(all_sparsities)
            variance = sum((x - mean) ** 2 for x in all_sparsities) / len(all_sparsities)
            overall_std = variance ** 0.5

            summary_lines.extend([
                f"Average Sparsity (all layers): {overall_avg:.6f}",
                f"Average Density (all layers): {1 - overall_avg:.6f}",
                f"Sparsity Range: [{overall_min:.6f}, {overall_max:.6f}]",
                f"Standard Deviation: {overall_std:.6f}",
                f"Total Samples: {len(all_sparsities)}",
            ])

        # QKV Shape Statistics
        summary_lines.extend([
            "",
            "QKV Shape Statistics:",
            "-" * 80,
        ])

        all_shapes = []
        for layer_idx, stats in self.layer_stats.items():
            if stats.get("qkv_shapes"):
                all_shapes.extend(stats["qkv_shapes"])

        if all_shapes:
            # Collect unique shapes
            q_shapes = set(tuple(s["Q"]) for s in all_shapes if "Q" in s)
            k_shapes = set(tuple(s["K"]) for s in all_shapes if "K" in s)
            v_shapes = set(tuple(s["V"]) for s in all_shapes if "V" in s)

            summary_lines.append(f"Q shapes seen: {sorted(q_shapes)}")
            summary_lines.append(f"K shapes seen: {sorted(k_shapes)}")
            summary_lines.append(f"V shapes seen: {sorted(v_shapes)}")

            # Show most recent shape
            last_shape = all_shapes[-1]
            summary_lines.append(f"Last Q shape: {last_shape.get('Q', 'N/A')}")
            summary_lines.append(f"Last K shape: {last_shape.get('K', 'N/A')}")
            summary_lines.append(f"Last V shape: {last_shape.get('V', 'N/A')}")
        else:
            summary_lines.append("No QKV shapes recorded")

        summary_lines.extend([
            "=" * 80,
            ""
        ])

        summary = "\n".join(summary_lines)

        # Write to file only
        with open(self.summary_file, "a", encoding="utf-8") as f:
            f.write(summary)

    def close(self):
        """Close log files and write final summary."""
        if self.sparsity_handle is not None:
            self.sparsity_handle.close()
            self.sparsity_handle = None

        # Write final summary
        self.write_summary()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "sparsity_handle") and self.sparsity_handle is not None:
            self.sparsity_handle.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
