#!/bin/bash

echo "### Starting Container ###"

# Check if models need to be downloaded
if [ "$DOWNLOAD_MODELS" = "True" ]; then
    echo "### Downloading models from $MODELS_URL ###"
    python setup/simply_download_pts.py "$MODELS_URL" "--mega"
    echo "### Models downloaded successfully ###"
else
    echo "### Skipping model download ###"
fi

# Run the main application or tests
if [ "$RUN_TESTS_ON_STARTUP" = "True" ]; then
    echo "### Running tests ###"
    python setup/simply_run_tests.py
else
    echo "### You can start using that UniversalLabeler enjoy :>  ###"
#    python main.py
fi
