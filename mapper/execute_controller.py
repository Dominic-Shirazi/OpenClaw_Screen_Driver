"""Skill execution controller for OCSD.

Handles loading a saved skill JSON, resolving entry/target nodes,
running the replay (via orchestrator or direct runner), and saving
the replay log.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.config import get_config

logger = logging.getLogger(__name__)


def cmd_execute(args: Any) -> int:
    """Executes a saved skill file.

    Args:
        args: Parsed CLI arguments (needs skill_file, target_label,
              skip_vlm, dry_run, step_mode, debug_ai).

    Returns:
        Process exit code (0 = success, 1 = failure).
    """
    from mapper.export import import_skill, load_skill_from_file
    from mapper.runner import run_skill

    # --skip-vlm overrides vlm_confirm in config at runtime
    if args.skip_vlm:
        cfg = get_config()
        cfg.setdefault("execution", {})["vlm_confirm"] = False
        logger.info("VLM validation disabled (--skip-vlm)")

    skill_path = Path(args.skill_file)
    if not skill_path.exists():
        logger.error("Skill file not found: %s", skill_path)
        return 1

    logger.info("Loading skill: %s", skill_path)
    skill_data = load_skill_from_file(skill_path)
    graph, metadata = import_skill(skill_data)

    logger.info(
        "Skill '%s': %d nodes, %d edges",
        metadata.get("name", "?"),
        graph.node_count,
        graph.edge_count,
    )

    # Determine entry node
    entry_id = skill_data.get("entry_node_id")
    if not entry_id:
        entry_nodes = graph.get_entry_nodes()
        if not entry_nodes:
            logger.error("No entry node found in skill")
            return 1
        entry_id = entry_nodes[0]

    # Determine target node
    if args.target_label:
        # Search nodes by label
        target_id = None
        for nid in graph.nodes:
            node_data = graph.get_node(nid)
            if node_data.get("label", "").lower() == args.target_label.lower():
                target_id = nid
                break
        if not target_id:
            logger.error("No node matching label '%s'", args.target_label)
            return 1
        logger.info("Target: '%s' (node %s)", args.target_label, target_id[:8])
    else:
        exit_nodes = graph.get_exit_nodes()
        if exit_nodes:
            target_id = exit_nodes[0]
        else:
            all_nodes = graph.nodes
            if not all_nodes:
                logger.error("Graph has no nodes")
                return 1
            target_id = all_nodes[-1]

    # Build step-through callback if --step or --debug-ai
    step_mode = getattr(args, "step_mode", False)
    debug_ai = getattr(args, "debug_ai", False)
    step_callback = None

    if step_mode or debug_ai:
        from mapper.runner import RunnerEventType

        def _step_event_handler(
            event_type: RunnerEventType, data: dict,
        ) -> None:
            """Event callback for --step and --debug-ai modes."""
            if debug_ai and event_type == RunnerEventType.ELEMENT_LOCATED:
                logger.info(
                    "  [debug-ai] Located via %s at (%s) conf=%.2f",
                    data.get("method", "?"),
                    data.get("point", "?"),
                    data.get("confidence", 0),
                )
            if step_mode and event_type == RunnerEventType.STEP_PREVIEW:
                node_id = data.get("node_id", "?")
                label = data.get("label", node_id[:8] if isinstance(node_id, str) else "?")
                try:
                    from recorder.step_ui import step_through_prompt
                    action = step_through_prompt(
                        label=label,
                        node_id=node_id,
                        step_num=data.get("step", 0) + 1,
                        action_type=data.get("element_type", ""),
                    )
                    if action == "abort":
                        raise KeyboardInterrupt("User aborted via step-through")
                except ImportError:
                    # Fallback to console prompt
                    resp = input(
                        f"\n  Step {data.get('step', 0) + 1}: "
                        f"{label} — [Enter]=execute, s=skip, q=abort: "
                    ).strip().lower()
                    if resp == "q":
                        raise KeyboardInterrupt("User aborted via step-through")

        step_callback = _step_event_handler

    # Run the skill — use orchestrator for full preflight + recovery
    use_orchestrator = not args.skip_vlm  # orchestrator needs VLM for recovery
    if use_orchestrator:
        from mapper.orchestrator import orchestrate_skill

        replay_log = orchestrate_skill(
            graph,
            start_id=entry_id,
            goal_id=target_id,
            dry_run=args.dry_run,
            skip_vlm=args.skip_vlm,
            event_callback=step_callback,
        )
    else:
        replay_log = run_skill(
            graph,
            start_id=entry_id,
            goal_id=target_id,
            dry_run=args.dry_run,
        )

    # Save replay log
    cfg = get_config()
    logs_dir = Path(cfg["paths"]["replay_logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{replay_log.replay_id}.json"
    with open(log_path, "w") as f:
        json.dump(replay_log.to_dict(), f, indent=2)
    logger.info("Replay log: %s", log_path)

    # Summary
    success_count = sum(1 for s in replay_log.steps if s.success)
    total = len(replay_log.steps)
    status = "SUCCESS" if replay_log.overall_success else "FAILED"
    logger.info(
        "Result: %s (%d/%d steps, %dms)",
        status, success_count, total, replay_log.duration_ms,
    )

    return 0 if replay_log.overall_success else 1
