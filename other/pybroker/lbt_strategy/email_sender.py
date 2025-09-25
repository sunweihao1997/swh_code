import smtplib
import email.utils
from email.mime.text import MIMEText
import credentials as creds # Import the secure credentials

def send_email_notification(subject: str, html_content: str):
    """Sends an email using the configuration from credentials.py."""
    
    print("Preparing to send email notification...")
    message = MIMEText(html_content, 'html', 'utf-8')
    message['To'] = email.utils.formataddr((creds.EMAIL_RECEIVER_NAME, creds.EMAIL_RECEIVER_ADDRESS))
    message['From'] = email.utils.formataddr((creds.EMAIL_SENDER_NAME, creds.EMAIL_SENDER_ADDRESS))
    message['Subject'] = subject
    
    server = None
    try:
        # Connect to the SMTP server using SSL
        server = smtplib.SMTP_SSL(creds.SMTP_SERVER, creds.SMTP_PORT)
        
        # Login to the email account
        server.login(creds.EMAIL_SENDER_ADDRESS, creds.EMAIL_SENDER_PASSWORD)
        
        # Send the email
        server.sendmail(creds.EMAIL_SENDER_ADDRESS, [creds.EMAIL_RECEIVER_ADDRESS], msg=message.as_string())
        
        print(f"Successfully sent email notification to '{creds.EMAIL_RECEIVER_ADDRESS}'")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")
    finally:
        if server:
            server.quit()
