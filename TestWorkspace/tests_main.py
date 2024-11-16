import subprocess
import os
from common.general_parameters import BASE_PROJECT_DIRECTORY_PATH

# List of test script paths
test_scripts = [
    os.path.join(BASE_PROJECT_DIRECTORY_PATH, "TestWorkspace", "test_universal_labler", "segmentation", "test.py"),
    os.path.join(BASE_PROJECT_DIRECTORY_PATH, "TestWorkspace", "test_universal_labler", "detection", "test.py"),
    os.path.join(BASE_PROJECT_DIRECTORY_PATH, "TestWorkspace", "test_factory_and_models", "segmentation", "test.py"),
    os.path.join(BASE_PROJECT_DIRECTORY_PATH, "TestWorkspace", "test_factory_and_models", "detection", "test.py"),
]

# Dictionary to track summary of results
summary = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "error": 0
}

def run_test_script(script_path):
    """Run a test script and print the result."""
    summary["total"] += 1
    try:
        result = subprocess.run(
            ["python3", script_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        status = "SUCCESS" if result.returncode == 0 else "FAILED"
        if status == "SUCCESS":
            summary["passed"] += 1
        else:
            summary["failed"] += 1

        # Print the test result details
        print(f"Test Script: {script_path}")
        print(f"Status: {status}")
        print("Output:")
        print(result.stdout)
        if result.returncode != 0:
            print("Errors:")
            print(result.stderr)
        print("="*80)

    except Exception as e:
        summary["error"] += 1
        print(f"Test Script: {script_path}")
        print(f"Status: ERROR")
        print(f"Error: {str(e)}")
        print("="*80)

def main():
    print("Running all test scripts...\n")
    for script in test_scripts:
        run_test_script(script)

    # Summary report
    print("\nSUMMARY REPORT")
    print("="*30)
    print(f"Total Tests Run: {summary['total']}")
    print(f"Tests Passed: {summary['passed']}")
    print(f"Tests Failed: {summary['failed']}")
    print(f"Tests with Errors: {summary['error']}")
    print("="*30)

if __name__ == "__main__":
    main()
