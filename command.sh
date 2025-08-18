#!/bin/bash

module load python/3.9

g++ main.cpp -o main \
$(python3-config --includes) \
-L$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-lpython3.9 \
-Wl,-rpath=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

TARGET_FILE="./main"
# Loop until the file exists
until [ -f "$TARGET_FILE" ]; do
    echo "Waiting for $TARGET_FILE to appear..."
    sleep 2 # Wait for 2 seconds before checking again
done

echo "running..."
./main