# --- Email Sender Configuration ---
# IMPORTANT: Use an "App Password" for Gmail or an "SMTP Authorization Code" for QQ Mail,
# NOT your main account password. This is more secure.
EMAIL_SENDER_ADDRESS = "2309598788@qq.com"  # The email address to send from
EMAIL_SENDER_PASSWORD = "aoyqtzjhzmxaeafg"   # The App Password / Authorization Code
EMAIL_SENDER_NAME = "Stock Screener Bot"     # The name that appears in the "From" field

# --- Email Receiver ---
EMAIL_RECEIVER_ADDRESS = "sunweihao97@gmail.com" # The email address to send the report to
EMAIL_RECEIVER_NAME = "sun"                      # The name of the recipient

# --- SMTP Server Details ---
# For QQ Mail:
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465 # Port for SSL

# For Gmail, you would use:
# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 465
