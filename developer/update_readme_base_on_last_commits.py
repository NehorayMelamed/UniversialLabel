import os
import subprocess

from common.general_parameters import BASE_PROJECT_DIRECTORY_PATH

def get_last_commits(count=3):
    """
    Retrieve the last few commits from the current git repository.

    Args:
        count (int): Number of commits to retrieve.

    Returns:
        list: A list of formatted commit messages with author and description only.
    """
    try:
        result = subprocess.run(
            ["git", "log", f"-{count}", "--pretty=format:%an - %s"],
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )
        commits = result.stdout.strip().split("\n")
        return commits
    except subprocess.CalledProcessError as e:
        print("Error retrieving commits:", e)
        return []


def append_to_readme(commits, readme_path=os.path.join(BASE_PROJECT_DIRECTORY_PATH, "readme.md")):
    """
    Append commit messages to the README.md under 'New and Last Updates'.

    Args:
        commits (list): A list of commit messages to append.
        readme_path (str): Path to the README.md file.
    """
    if not os.path.exists(readme_path):
        print(f"README file not found at {readme_path}")
        return

    with open(readme_path, "r") as file:
        lines = file.readlines()

    # Locate or create the "New and Last Updates" section
    section_header = "## New and Last Updates\n"
    try:
        start_index = lines.index(section_header) + 1
    except ValueError:
        # Section not found, append at the end
        lines.append("\n" + section_header)
        start_index = len(lines)

    # Add the commits to the section
    for commit in commits:
        lines.insert(start_index, f"- {commit}\n")
        start_index += 1

    # Write back the changes to the README.md
    with open(readme_path, "w") as file:
        file.writelines(lines)

    print(f"Updated README.md with the latest {len(commits)} commits.")


if __name__ == "__main__":
    # Get the last 3 commits
    last_commits = get_last_commits(count=3)

    if last_commits:
        # Update the README.md
        append_to_readme(last_commits)
    else:
        print("No commits found or failed to retrieve commits.")
