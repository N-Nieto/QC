import sys
from pathlib import Path


def ensure_dir(target_path: str, emergency_base: str = "emergency_dir") -> None:
    """
    Ensure a directory exists at `target_path`. If creation fails,
    fall back to an emergency directory in the current working directory.

    Args:
        target_path (str): Desired absolute/relative directory path.
        emergency_base (str): Base name for emergency directory (default: "emergency_dir").

    Returns:
        str: Path to the ensured directory (either target or emergency).
    """
    target_path = Path(target_path).absolute()  # Convert to absolute path
    emergency_dir = Path.cwd() / emergency_base  # Emergency dir in current working dir

    if target_path.exists():
        print(f"Directory already exists at: '{target_path}'")
        return

    try:
        # Attempt to create target directory
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created at: '{target_path}'")
        return
    except (OSError, PermissionError) as e:
        # Fallback: Create emergency directory
        print(
            f"Failed to create directory at '{target_path}': {e}\n"
            f"Using emergency directory: '{emergency_dir}'",
            file=sys.stderr,
        )
        emergency_dir.mkdir(exist_ok=True)  # Safe by design (current dir is writable)
        if emergency_dir.exists():
            print(f"Emergency directory successfully created at: '{emergency_dir}'")
        return
