"""Standalone script to generate and display/save Sokoban puzzles.
Bypasses ragen/__init__.py to avoid torch/CUDA imports on login nodes.

Usage:
  # Display mode
  python generate_puzzles.py --dim 6 --steps 4 --count 10

  # Save mode (train + val parquets with dedup)
  python generate_puzzles.py --dim 6 --steps 4 --train-count 5000 --val-count 1000 --save-dir data/sokoban_4step/

  # Parallel save mode
  python generate_puzzles.py --dim 6 --steps 4 --train-count 5000 --val-count 1000 --save-dir data/sokoban_4step/ --workers 8
"""
import importlib.util
import numpy as np
import argparse
import json
import os
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Direct import of utils to avoid ragen/__init__.py (which pulls in torch)
spec = importlib.util.spec_from_file_location(
    "sokoban_utils",
    os.path.join(os.path.dirname(__file__), "ragen/env/sokoban/utils.py")
)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

GRID_LOOKUP = {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"}
ACTION_NAMES = {1: "Up", 2: "Down", 3: "Left", 4: "Right"}


def _puzzle_hash(room_state, room_fixed):
    """Hash a puzzle by its state + structure arrays for dedup."""
    return hashlib.md5(room_state.tobytes() + room_fixed.tobytes()).hexdigest()


def _render_grid(room_state, room_structure):
    room_display = np.where(
        (room_state == 5) & (room_structure == 2), 6, room_state
    )
    return "\n".join(
        "".join(GRID_LOOKUP.get(int(cell), "?") for cell in row)
        for row in room_display
    )


def _generate_single_puzzle(args_tuple):
    """Generate a single puzzle. Used by both serial and parallel modes."""
    dim, num_boxes, max_solution_length, search_depth, num_steps, nonlinear, linear = args_tuple
    try:
        room_structure, room_state, box_mapping, _ = utils.generate_room(
            dim=dim,
            num_boxes=num_boxes,
            max_solution_length=max_solution_length,
            search_depth=search_depth,
            num_steps=num_steps,
        )
        shortest = utils.get_shortest_action_path(
            room_structure, room_state, MAX_DEPTH=max_solution_length + 10
        )
        if nonlinear and len(set(shortest)) <= 1:
            return None, None  # reject linear puzzles
        if linear and len(set(shortest)) > 1:
            return None, None  # reject nonlinear puzzles
        h = _puzzle_hash(room_state, room_structure)
        puzzle = {
            "room_fixed": json.dumps(room_structure.tolist()),
            "room_state": json.dumps(room_state.tolist()),
            "box_mapping": json.dumps({str(k): [int(x) for x in v] for k, v in box_mapping.items()}),
            "solution_length": len(shortest),
            "solution": json.dumps([int(a) for a in shortest]),
            "dim": json.dumps(list(dim)),
        }
        return puzzle, h
    except (RuntimeError, RuntimeWarning):
        return None, None


def generate_puzzles(dim, num_boxes, max_solution_length, num_puzzles,
                     search_depth=300, num_steps=100, exclude_hashes=None,
                     workers=1, desc="Generating", nonlinear=False, linear=False):
    """Generate puzzles with optional parallelism. Deduplicates against exclude_hashes."""
    if exclude_hashes is None:
        exclude_hashes = set()

    puzzles = []
    seen = set(exclude_hashes)
    args_tuple = (dim, num_boxes, max_solution_length, search_depth, num_steps, nonlinear, linear)

    if workers > 1:
        # Parallel mode: submit batches of work
        pbar = tqdm(total=num_puzzles, desc=desc)
        dupes = 0
        failures = 0
        max_total_attempts = num_puzzles * 200

        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = set()
            submitted = 0

            # Keep submitting work until we have enough puzzles
            while len(puzzles) < num_puzzles and submitted < max_total_attempts:
                # Keep pool full
                while len(pending) < workers * 4 and submitted < max_total_attempts:
                    future = executor.submit(_generate_single_puzzle, args_tuple)
                    pending.add(future)
                    submitted += 1

                # Collect completed futures
                done = {f for f in pending if f.done()}
                if not done:
                    # Wait for at least one
                    done_iter = as_completed(pending)
                    done = {next(done_iter)}

                for future in done:
                    pending.discard(future)
                    puzzle, h = future.result()
                    if puzzle is None:
                        failures += 1
                        continue
                    if h in seen:
                        dupes += 1
                        continue
                    seen.add(h)
                    puzzles.append(puzzle)
                    pbar.update(1)
                    pbar.set_postfix(dupes=dupes, failures=failures)
                    if len(puzzles) >= num_puzzles:
                        break

            # Cancel remaining
            for f in pending:
                f.cancel()

        pbar.close()
    else:
        # Serial mode
        attempts = 0
        max_attempts = num_puzzles * 200
        dupes = 0
        failures = 0
        pbar = tqdm(total=num_puzzles, desc=desc)

        while len(puzzles) < num_puzzles and attempts < max_attempts:
            attempts += 1
            puzzle, h = _generate_single_puzzle(args_tuple)
            if puzzle is None:
                failures += 1
                continue
            if h in seen:
                dupes += 1
                continue
            seen.add(h)
            puzzles.append(puzzle)
            pbar.update(1)
            pbar.set_postfix(dupes=dupes, failures=failures)

        pbar.close()

    if len(puzzles) < num_puzzles:
        print(f"[Warning] Only generated {len(puzzles)}/{num_puzzles} puzzles")

    print(f"  Done: {len(puzzles)} puzzles, {dupes} duplicates skipped, {failures} generation failures")
    return puzzles, seen


def display_puzzles(dim, num_boxes, max_solution_length, num_puzzles,
                    search_depth=300, num_steps=100, workers=1, nonlinear=False, linear=False):
    """Generate and print puzzles to console."""
    puzzles, _ = generate_puzzles(
        dim, num_boxes, max_solution_length, num_puzzles, search_depth, num_steps,
        workers=workers, desc="Generating puzzles", nonlinear=nonlinear, linear=linear
    )
    for i, p in enumerate(puzzles):
        room_state = np.array(json.loads(p["room_state"]))
        room_structure = np.array(json.loads(p["room_fixed"]))
        solution = json.loads(p["solution"])

        grid = _render_grid(room_state, room_structure)
        player = tuple(np.argwhere(room_state == 5)[0])
        boxes = [tuple(pos) for pos in np.argwhere(room_state == 4)]
        targets = [tuple(pos) for pos in np.argwhere(room_structure == 2)]

        print(f"=== Puzzle {i+1} === (BFS solution: {p['solution_length']} steps)")
        print(f"Grid: {dim[0]}x{dim[1]}, Boxes: {num_boxes}")
        print(f"Player: {player}, Boxes: {boxes}, Targets: {targets}")
        print(grid)
        print(f"Solution: {' || '.join(ACTION_NAMES[a] for a in solution)}")
        print()


def save_puzzles(dim, num_boxes, max_solution_length, train_count, val_count,
                 save_dir, search_depth=300, num_steps=100, workers=1, nonlinear=False, linear=False):
    """Generate and save train/val parquet files with dedup."""
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    # Generate val set first
    val_puzzles, val_hashes = generate_puzzles(
        dim, num_boxes, max_solution_length, val_count, search_depth, num_steps,
        workers=workers, desc="Generating val puzzles", nonlinear=nonlinear, linear=linear
    )

    # Generate train set, excluding val puzzles
    train_puzzles, _ = generate_puzzles(
        dim, num_boxes, max_solution_length, train_count, search_depth, num_steps,
        exclude_hashes=val_hashes, workers=workers, desc="Generating train puzzles",
        nonlinear=nonlinear, linear=linear
    )

    # Save as parquet
    train_path = os.path.join(save_dir, "train.parquet")
    val_path = os.path.join(save_dir, "val.parquet")

    pd.DataFrame(train_puzzles).to_parquet(train_path, index=False)
    pd.DataFrame(val_puzzles).to_parquet(val_path, index=False)

    val_json_path = os.path.join(save_dir, "val.json")
    with open(val_json_path, "w") as f:
        json.dump(val_puzzles, f, indent=2)

    # Save human-readable preview of first 5 val puzzles
    preview_path = os.path.join(save_dir, "val_preview.txt")
    with open(preview_path, "w") as f:
        for i, p in enumerate(val_puzzles[:5]):
            room_state = np.array(json.loads(p["room_state"]))
            room_structure = np.array(json.loads(p["room_fixed"]))
            solution = json.loads(p["solution"])

            grid = _render_grid(room_state, room_structure)
            player = tuple(np.argwhere(room_state == 5)[0]) if 5 in room_state else tuple(np.argwhere(
                np.where((room_state == 5) & (room_structure == 2), 6, room_state) == 6)[0])
            boxes = [tuple(pos) for pos in np.argwhere(room_state == 4)]
            targets = [tuple(pos) for pos in np.argwhere(room_structure == 2)]

            f.write(f"=== Puzzle {i+1} === (Solution: {p['solution_length']} steps)\n")
            f.write(f"Player: {player}, Boxes: {boxes}, Targets: {targets}\n")
            f.write(grid + "\n")
            f.write(f"Solution: {' || '.join(ACTION_NAMES[a] for a in solution)}\n")
            f.write("\n")

    print(f"\nSaved {len(train_puzzles)} train puzzles to {train_path}")
    print(f"Saved {len(val_puzzles)} val puzzles to {val_path}")
    print(f"Saved {len(val_puzzles)} val puzzles to {val_json_path}")
    print(f"Saved preview of 5 val puzzles to {preview_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sokoban puzzles")
    parser.add_argument("--dim", type=int, default=20, help="Grid dimension (square)")
    parser.add_argument("--boxes", type=int, default=1, help="Number of boxes")
    parser.add_argument("--steps", type=int, default=8, help="Exact solution length")
    parser.add_argument("--count", type=int, default=10, help="Number of puzzles (display mode)")
    parser.add_argument("--search-depth", type=int, default=300, help="Search depth for room generation")
    parser.add_argument("--num-steps", type=int, default=100, help="Random walk steps for topology (more = fewer walls)")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save train/val parquets")
    parser.add_argument("--train-count", type=int, default=5000, help="Number of train puzzles (save mode)")
    parser.add_argument("--val-count", type=int, default=1000, help="Number of val puzzles (save mode)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (1 = serial)")
    parser.add_argument("--nonlinear", action="store_true", help="Only generate puzzles with mixed directions (no straight-line pushes)")
    parser.add_argument("--linear", action="store_true", help="Only generate linear puzzles (all pushes same direction)")
    args = parser.parse_args()

    if args.nonlinear and args.linear:
        parser.error("--nonlinear and --linear are mutually exclusive")

    dim = (args.dim, args.dim)

    if args.save_dir:
        save_puzzles(
            dim=dim,
            num_boxes=args.boxes,
            max_solution_length=args.steps,
            train_count=args.train_count,
            val_count=args.val_count,
            save_dir=args.save_dir,
            search_depth=args.search_depth,
            num_steps=args.num_steps,
            workers=args.workers,
            nonlinear=args.nonlinear,
            linear=args.linear,
        )
    else:
        display_puzzles(
            dim=dim,
            num_boxes=args.boxes,
            max_solution_length=args.steps,
            num_puzzles=args.count,
            search_depth=args.search_depth,
            num_steps=args.num_steps,
            workers=args.workers,
            nonlinear=args.nonlinear,
            linear=args.linear,
        )
