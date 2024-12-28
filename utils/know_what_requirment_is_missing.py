import pkg_resources
import sys

def check_missing_requirements(requirements_file: str):
    """
    Check for missing libraries in the current environment based on a requirements.txt file.

    Args:
        requirements_file (str): Path to the requirements.txt file.

    Returns:
        list: List of missing libraries.
    """
    try:
        # Read the requirements file
        with open(requirements_file, "r") as f:
            requirements = f.readlines()

        # Parse the requirements
        requirements = [line.strip() for line in requirements if line.strip() and not line.startswith("#")]

        # Check for missing libraries
        missing = []
        for requirement in requirements:
            try:
                pkg_resources.require(requirement)
            except pkg_resources.DistributionNotFound:
                missing.append(requirement)
            except pkg_resources.VersionConflict as e:
                print(f"Version conflict for {requirement}: {e}")

        return missing

    except FileNotFoundError:
        print(f"Error: File not found: {requirements_file}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":

    requirements_file = "/home/nehoray/PycharmProjects/tests_wotkspace/Segment-Everything-Everywhere-All-At-Once/assets/requirements/requirements.txt"
    missing_packages = check_missing_requirements(requirements_file)

    if missing_packages:
        print("\nMissing packages:")
        for package in missing_packages:
            print(f"  - {package}")
    else:
        print("\nAll required packages are installed.")
