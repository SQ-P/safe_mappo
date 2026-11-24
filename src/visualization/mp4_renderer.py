"""使用 Matplotlib 绘制回放帧并通过 ImageIO/FFmpeg 编码 MP4。"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from src.utils.io_utils import ensure_dir


def _agent_color(agent_id: int):
    """从 tab10 循环取色，区分相邻智能体。"""
    cmap = plt.get_cmap("tab10")
    return cmap(int(agent_id) % 10)


def _draw_frame(frame, map_size, goal_radius: float, agent_radius: float) -> np.ndarray:
    """绘制实际轨迹、当前子目标及其与最终目标的关系。"""
    figure = plt.figure(figsize=(11, 6), dpi=100)
    layout = figure.add_gridspec(1, 2, width_ratios=(1.75, 1), wspace=0.12)
    axis = figure.add_subplot(layout[0, 0])
    panel = figure.add_subplot(layout[0, 1])
    panel.axis("off")
    axis.set_xlim(0, float(map_size[0]))
    axis.set_ylim(0, float(map_size[1]))
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor("#f7f7f7")
    axis.grid(color="#dddddd", linewidth=0.5, alpha=0.7)

    goals = np.asarray(frame.goal, dtype=np.float32).reshape(-1, 2)
    for goal_id, goal in enumerate(goals):
        axis.add_patch(Circle(goal, goal_radius, facecolor="#7bc96f", edgecolor="#238b45", alpha=0.35))
        axis.plot(goal[0], goal[1], marker="*", color="#006d2c", markersize=10)
        goal_label_x = goal[0] - 4.0 if goal[0] > map_size[0] - 15 else goal[0] + 1.5
        axis.text(goal_label_x, goal[1] - 5.0, f"G{goal_id}", color="#006d2c", fontsize=8)

    for position, size, velocity in zip(
        frame.obstacle_positions,
        frame.obstacle_sizes,
        frame.obstacle_velocities,
    ):
        radius = float(size[0]) / 2.0
        axis.add_patch(Circle(position, radius, facecolor="#666666", edgecolor="#222222", alpha=0.8))
        # 静态障碍时速度为0，因此不会显示多余箭头。
        if float(np.linalg.norm(velocity)) > 1e-8:
            axis.arrow(
                position[0], position[1], velocity[0] * 3.0, velocity[1] * 3.0,
                width=0.15,
                head_width=1.0,
                color="#fdd835",
                length_includes_head=True,
            )

    if frame.oracle_subgoal_enabled:
        for agent_id, (position, subgoal) in enumerate(
            zip(frame.agent_positions, frame.subgoals)
        ):
            color = _agent_color(agent_id)
            # This line shows the active guidance target, not a predicted path.
            axis.plot(
                [position[0], subgoal[0]], [position[1], subgoal[1]],
                color=color, linestyle="--", linewidth=1.7, alpha=0.85,
                zorder=3,
            )
            axis.add_patch(Circle(
                subgoal, frame.subgoal_reach_radius,
                facecolor=color, edgecolor=color, linewidth=1.1,
                linestyle=":", alpha=0.27, zorder=4,
            ))
            axis.scatter(
                subgoal[0], subgoal[1], marker="D", s=75,
                facecolor=color, edgecolor="white", linewidth=1.1,
                zorder=6,
            )
            subgoal_label_x = (
                subgoal[0] - 12.0
                if subgoal[0] > map_size[0] - 15
                else subgoal[0] + 1.5
            )
            axis.text(
                subgoal_label_x, subgoal[1] + 3.0,
                f"SG{agent_id}", color=color, fontsize=9,
                fontweight="bold", zorder=7,
            )
            if (
                frame.subgoal_versions[agent_id] > 0
                and 0 <= frame.step - frame.last_subgoal_update_steps[agent_id] <= 9
            ):
                old = frame.previous_subgoals[agent_id]
                axis.scatter(
                    old[0], old[1], marker="x", s=55,
                    color=color, alpha=0.45, linewidth=1.5, zorder=5,
                )

    for agent_id, (position, path) in enumerate(
        zip(frame.agent_positions, frame.paths)
    ):
        color = _agent_color(agent_id)
        if len(path) > 1:
            axis.plot(
                path[:, 0],
                path[:, 1],
                color=color,
                linewidth=1.5,
                alpha=0.75,
            )
        axis.add_patch(
            Circle(
                position,
                agent_radius,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
            )
        )
        agent_label_x = position[0] - 8 if position[0] > map_size[0] - 25 else position[0] + 2
        axis.text(
            agent_label_x,
            position[1] - 4,
            f"A{agent_id}",
            color=color,
            fontsize=8,
        )

    status_color = (
        "#238b45"
        if frame.reason == "team_success"
        else "#b22222"
        if frame.terminated
        else "#333333"
    )
    figure.suptitle(
        f"step={frame.step}  return={frame.team_return:.3f}  "
        f"required_goal_distance={frame.required_goal_distance:.1f}  "
        f"status={frame.reason}",
        color=status_color,
        fontsize=11,
    )
    panel.text(0.02, 0.98, "Subgoal guidance", fontsize=14,
               fontweight="bold", va="top")
    panel.text(0.02, 0.89, "Solid trail: flown path", fontsize=9, va="top")
    panel.text(0.02, 0.84, "Diamond / ring: active SG / reach zone", fontsize=9, va="top")
    panel.text(0.02, 0.79, "Dashed: UAV to SG   Green star: final goal", fontsize=9, va="top")
    panel.text(0.02, 0.74, "Guide lines are not predicted trajectories.",
               fontsize=8, color="#555555", va="top")
    panel.text(0.02, 0.68, f"Planner calls: {frame.subgoal_replan_count}",
               fontsize=9, va="top")

    if frame.oracle_subgoal_enabled:
        row_gap = min(0.16, 0.46 / max(len(frame.subgoals), 1))
        for agent_id, (position, subgoal) in enumerate(
            zip(frame.agent_positions, frame.subgoals)
        ):
            y = 0.61 - agent_id * row_gap
            color = _agent_color(agent_id)
            distance = float(np.linalg.norm(position - subgoal))
            panel.text(
                0.02, y,
                f"A{agent_id}  SG{agent_id} #{frame.subgoal_versions[agent_id]}"
                f"  ({subgoal[0]:.1f}, {subgoal[1]:.1f})",
                color=color, fontsize=10, fontweight="bold", va="top",
            )
            events = [f"d={distance:.1f} m"]
            if (
                frame.subgoal_versions[agent_id] > 0
                and 0 <= frame.step - frame.last_subgoal_update_steps[agent_id] <= 9
            ):
                events.append(f"new @{frame.last_subgoal_update_steps[agent_id]}")
            if 0 <= frame.step - frame.last_subgoal_reach_steps[agent_id] <= 9:
                events.append(f"prior hit @{frame.last_subgoal_reach_steps[agent_id]}")
            panel.text(0.05, y - 0.055, "  |  ".join(events),
                       color="#333333", fontsize=8.5, va="top")
    else:
        panel.text(0.02, 0.58, "Oracle subgoals disabled", fontsize=10,
                   color="#555555", va="top")

    panel.text(
        0.02, 0.12,
        f"Last transition SG reward: {frame.subgoal_progress_reward:+.4f}",
        fontsize=9, va="top",
    )
    panel.text(
        0.02, 0.07,
        "On a replan step, reward used the prior SG.",
        fontsize=8, color="#555555", va="top",
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.10)
    figure.canvas.draw()
    image = np.asarray(
        figure.canvas.buffer_rgba(), dtype=np.uint8
    )[..., :3].copy()
    plt.close(figure)
    return image


def render_episode_mp4(
    episode,
    output_path: str | Path,
    map_size,
    goal_radius: float,
    agent_radius: float,
    fps: int = 20,
) -> Path:
    """把一轮不可变回放编码为 H.264 MP4。"""
    fps = int(fps)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not episode.frames:
        raise ValueError("episode must contain at least one frame")

    output = Path(output_path)
    ensure_dir(output.parent)
    writer = None
    try:
        writer = imageio.get_writer(
            output,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            macro_block_size=None,
        )
        for frame in episode.frames:
            writer.append_data(
                _draw_frame(frame, map_size, goal_radius, agent_radius)
            )

        terminal_image = _draw_frame(
            episode.frames[-1], map_size, goal_radius, agent_radius
        )
        for _ in range(fps):
            writer.append_data(terminal_image)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "MP4 encoding requires FFmpeg support; install it with "
            "'python -m pip install imageio-ffmpeg'."
        ) from exc
    finally:
        if writer is not None:
            writer.close()

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("FFmpeg did not create a non-empty MP4 file")
    return output
