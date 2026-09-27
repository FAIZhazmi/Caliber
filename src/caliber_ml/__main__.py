"""Package entry point."""

from pathlib import Path

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project


def main(*args, **kwargs):
    """Run the project's Kedro command."""
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    return find_run_command(package_name)(*args, **kwargs)


if __name__ == "__main__":
    main()

