from __future__ import annotations

import argparse
import json
from pathlib import Path

import pygame


COLORS = {
    "#": (39, 39, 42),
    ".": (244, 244, 245),
    "P": (234, 179, 8),
    "D": (220, 38, 38),
    "d": (34, 197, 94),
    "K": (250, 204, 21),
    "G": (168, 85, 247),
    "T": (127, 29, 29),
    "A": (37, 99, 235),
    "B": (249, 115, 22),
}


def load_replay(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw(screen, font, frame, cell_size: int, side_width: int) -> None:
    level = [list(row) for row in frame["level"]]
    for r, c in frame.get("doors", []):
        level[r][c] = "d" if frame["door_open"] else "D"
    key = frame.get("key")
    if key and frame["key_taken"]:
        level[key[0]][key[1]] = "."
    for agent, pos in frame["agents"].items():
        level[pos[0]][pos[1]] = "A" if agent == "agent_0" else "B"

    screen.fill((250, 250, 250))
    for r, row in enumerate(level):
        for c, cell in enumerate(row):
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, COLORS.get(cell, COLORS["."]), rect)
            pygame.draw.rect(screen, (212, 212, 216), rect, 1)
            if cell in {"A", "B", "K", "G", "P"}:
                label = font.render(cell, True, (24, 24, 27))
                screen.blit(label, label.get_rect(center=rect.center))

    panel_x = len(level[0]) * cell_size + 20
    lines = [
        "Coop Puzzle MARL",
        f"Step: {frame['step']}",
        f"Reward: {frame['reward']:.2f}",
        f"Door: {'Open' if frame['door_open'] else 'Closed'}",
        f"Key: {'Taken' if frame['key_taken'] else 'Waiting'}",
        f"Success: {'Yes' if frame.get('success') else 'No'}",
        "",
        "A: blue agent",
        "B: orange agent",
        "P: pressure plate",
        "D: door",
        "K: key",
        "G: goal",
    ]
    for i, text in enumerate(lines):
        color = (24, 24, 27) if i != 0 else (15, 23, 42)
        surf = font.render(text, True, color)
        screen.blit(surf, (panel_x, 24 + i * 26))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved cooperative puzzle trajectory.")
    parser.add_argument("--replay", default="outputs/eval/rule_replay.json")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--cell-size", type=int, default=64)
    args = parser.parse_args()
    frames = load_replay(args.replay)
    if not frames:
        raise ValueError("Replay is empty.")

    pygame.init()
    rows = len(frames[0]["level"])
    cols = len(frames[0]["level"][0])
    side_width = 300
    screen = pygame.display.set_mode((cols * args.cell_size + side_width, rows * args.cell_size))
    pygame.display.set_caption("Cooperative Puzzle MARL Replay")
    font = pygame.font.SysFont("consolas", 20)
    clock = pygame.time.Clock()
    index = 0
    paused = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    index = min(index + 1, len(frames) - 1)
                elif event.key == pygame.K_LEFT:
                    index = max(index - 1, 0)
                elif event.key == pygame.K_r:
                    index = 0
        draw(screen, font, frames[index], args.cell_size, side_width)
        pygame.display.flip()
        if not paused:
            index = (index + 1) % len(frames)
        clock.tick(args.fps)
    pygame.quit()


if __name__ == "__main__":
    main()

