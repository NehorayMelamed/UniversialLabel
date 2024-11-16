import os
import zipfile
from datetime import datetime

# Specify the base directory of your project
BASE_PROJECT_DIRECTORY_PATH = "/home/nehoray/PycharmProjects/UniversaLabeler"

# Create a backup function to zip all Python scripts
def backup_python_scripts(base_path):
    # Get the current date and time for a unique backup name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"python_scripts_backup_{timestamp}.zip"

    # Full path for the backup file
    backup_path = os.path.join(base_path, backup_filename)

    # Create a ZipFile object to write the Python files
    with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
        # Walk through all directories in the base path
        for foldername, subfolders, filenames in os.walk(base_path):
            for filename in filenames:
                # Only add Python scripts to the backup
                if filename.endswith('.py'):
                    file_path = os.path.join(foldername, filename)
                    # Add the Python script to the zip, preserving its path within the project
                    backup_zip.write(file_path, os.path.relpath(file_path, base_path))
                    print(f"Added {file_path} to backup.")

    print(f"Backup completed successfully: {backup_path}")

# Run the backup function
if __name__ == "__main__":
    backup_python_scripts(BASE_PROJECT_DIRECTORY_PATH)
