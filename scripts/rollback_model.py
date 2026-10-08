"""
Restores models/production_model from a given commit, for when a
promoted model needs to be rolled back. Does not commit or push: prints
the exact commands and stops, same policy as src/live/promote.py.

Run with:
    python scripts/rollback_model.py <commit-sha>
"""

import subprocess
import sys

TARGET = "models/production_model"


def rollback(commit_sha):
    subprocess.run(["git", "checkout", commit_sha, "--", TARGET], check=True)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <commit-sha>")
        sys.exit(1)

    commit_sha = sys.argv[1]
    print(f"Restoring {TARGET} from commit {commit_sha}...")
    rollback(commit_sha)
    print(f"Restored. {TARGET} now matches {commit_sha}.")
    print("To commit and push this rollback (not run automatically):")
    print(f"  git add {TARGET}")
    print(f'  git commit -m "Rollback model to {commit_sha}"')
    print("  git push")


if __name__ == "__main__":
    main()
