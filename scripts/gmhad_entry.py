"""Launch the pinned GM-HAD program on one prepared scene in a fresh process."""

import argparse
import ast
import os
import sys
from pathlib import Path

from scripts.comparison_helpers import enable_deterministic_cuda


def use_sorted_segment_median(tree):
    # GM-HAD uses the median value only. CUDA's indexed median is rejected
    # in strict mode. Sorting and selecting the lower middle value is equivalent.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "bg_soft_weights_from_segment":
            for statement in node.body:
                if (isinstance(statement, ast.Assign)
                        and ast.unparse(statement.value) == "x.median(dim=1, keepdim=True).values"):
                    statement.value = ast.parse(
                        "x.sort(dim=1).values.select(1, (x.size(1) - 1) // 2).unsqueeze(1)", mode="eval"
                    ).body
                    return
    raise ValueError("Unexpected GM-HAD segment median. Check the pinned revision in README.md.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--deterministic-cuda", action="store_true")
    args = parser.parse_args()
    source, scene, output = (path.resolve() for path in (args.source, args.scene, args.output))
    script = source / "main.py"
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    entry = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"), None)
    if entry is None:
        parser.error(f"Missing GM-HAD entry point in {script}. Check the pinned revision in README.md.")

    settings = {"dataset_list": [str(scene)], "result_dir": str(output)}
    replaced = set()
    for statement in entry.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in settings:
            statement.value = ast.parse(repr(settings[target.id]), mode="eval").body
            replaced.add(target.id)
    if replaced != settings.keys():
        parser.error(f"Unexpected GM-HAD configuration in {script}. Check the pinned revision in README.md.")

    if args.deterministic_cuda:
        enable_deterministic_cuda()
        use_sorted_segment_median(tree)
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.argv = [str(script)]
    namespace = {"__name__": "__main__", "__file__": str(script)}
    exec(compile(ast.fix_missing_locations(tree), str(script), "exec"), namespace)


if __name__ == "__main__":
    main()
