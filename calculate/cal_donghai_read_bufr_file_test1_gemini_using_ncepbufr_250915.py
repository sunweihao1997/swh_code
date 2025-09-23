import ncepbufr

# Define the path to your BUFR file
# You'll need to replace this with an actual file path.
bufr_file_path = '/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr'

try:
    # Open the BUFR file for reading ('r')
    bufr = ncepbufr.open(bufr_file_path, 'r')

    # Read the first message from the file
    # .read_msg() returns None when it reaches the end of the file
    msg = bufr.read_msg()

    if msg:
        print("Successfully read the first message!")
        print(f"  - Message Type: {msg.msg_type}")
        print(f"  - Message Subtype: {msg.msg_subtype}")
        # Each message contains a set of data descriptors, or 'mnemonics'
        # Let's print the first 10 mnemonics available in this message
        print(f"  - First 10 mnemonics: {msg.mnemonics[:10]}")
    else:
        print("Could not read a message. The file might be empty or invalid.")

    # It's important to close the file when you are done
    bufr.close()

except FileNotFoundError:
    print(f"Error: The file was not found at '{bufr_file_path}'")
except Exception as e:
    print(f"An error occurred: {e}")