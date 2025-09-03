#!/bin/bash

module load python/3.10
source pyROMenv/bin/activate

g++ main.cpp -o main \
$(python3-config --includes) \
-L$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-lpython3.10 \
-Wl,-rpath=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

TARGET_FILE="./main"
# Loop until the file exists
until [ -f "$TARGET_FILE" ]; do
    echo "Waiting for $TARGET_FILE to appear..."
    sleep 2 # Wait for 2 seconds before checking again
done

echo "running..."
./main

deactivate