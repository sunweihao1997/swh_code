import subprocess
import os
import re

# --- Configuration ---
# Path to your satellite BUFR file
bufr_file_path = '/mnt/f/data/1bamua/1bamua_20250424/gdas.1bamua.t12z.20250424.bufr'
# Name for the temporary decoded text file
decoded_output_file = 'decoded_bufr_contents.txt'
# Set the maximum subsets limit higher than what binv reported (386173)
max_subsets = 400000

# --- Step 1: Decode the large BUFR file using the 'debufr' utility ---
print("Decoding large BUFR file with the 'debufr' utility. This may take a moment...")

try:
    # Construct the command to run. The -p flag sets the memory parameter. [1]
    command =
    
    # Execute the command
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(f"Successfully decoded BUFR file to '{decoded_output_file}'")

except FileNotFoundError:
    print("Error: 'debufr' command not found.")
    print("Please ensure that the NCEPLIBS-bufr bin directory is in your system's PATH.")
    exit()
except subprocess.CalledProcessError as e:
    print("Error running the 'debufr' command:")
    print(e.stderr)
    exit()

# --- Step 2: Parse the decoded text file to find the required information ---
print("\nParsing decoded file for satellite information...")

satellite_id = None
platform_id = None

try:
    with open(decoded_output_file, 'r') as f:
        for line in f:
            # Use regular expressions to find lines with SAID or BPID and extract the value
            if ' SAID ' in line:
                # Find the first number (integer or float) on the line
                match = re.search(r'\s+([0-9.]+)\s+', line)
                if match and satellite_id is None: # Store the first one we find
                    satellite_id = match.group(1)
            
            if ' BPID ' in line:
                match = re.search(r'\s+([0-9.]+)\s+', line)
                if match and platform_id is None: # Store the first one we find
                    platform_id = match.group(1)

            # If we've found both, we can stop parsing
            if satellite_id and platform_id:
                break

    # --- Step 3: Display the results ---
    print("\n--- Extracted Information ---")
    if satellite_id:
        print(f"Satellite Identifier (SAID): {satellite_id}")
    else:
        print("Satellite Identifier (SAID): Not found in the file.")

    if platform_id:
        print(f"Platform Identifier (BPID): {platform_id}")
    else:
        print("Platform Identifier (BPID): Not found in the file.")
    print("---------------------------\n")


except FileNotFoundError:
    print(f"Error: The output file '{decoded_output_file}' was not created.")
finally:
    # --- Step 4: Clean up the temporary file ---
    if os.path.exists(decoded_output_file):
        os.remove(decoded_output_file)
        print(f"Cleaned up temporary file: '{decoded_output_file}'")