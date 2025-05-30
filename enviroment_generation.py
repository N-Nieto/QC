# %%
import os
import subprocess
import yaml


def export_environment_to_yml(output_file="environment.yml"):
    """Export the current Python environment to a YAML file."""
    try:
        # Ensure we are in a virtual environment
        venv_path = os.getenv('VIRTUAL_ENV')
        if not venv_path:
            print("Warning: You are not in a virtual environment.")
            venv_path = "None"

        # Get installed packages
        result = subprocess.run(["pip", "freeze"], check=True,
                                capture_output=True, text=True)
        packages = result.stdout.strip().splitlines()

        # Prepare YAML structure
        environment_data = {
            "name": "my_environment",
            "dependencies": packages,
        }

        # Write to YAML file
        with open(output_file, "w") as yml_file:
            yaml.dump(environment_data, yml_file, default_flow_style=False)

        print(f"Environment exported to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during export: {e}")


export_environment_to_yml()
# %%
