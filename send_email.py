import smtplib
from email.mime.text import MIMEText

import os

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

def go_mail(subject, body, attachment_path, sender, recipients, password):

    with open(attachment_path, "rb") as attachment:
        # Add the attachment to the message
        part = MIMEBase("application", "octet-stream")
        part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {os.path.basename('usg_analysis.jpg')}",

    )


    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    html_part = MIMEText(body)
    msg.attach(html_part)
    msg.attach(part)


    

    smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    smtp_server.login(sender, password)
    smtp_server.sendmail(sender, recipients, msg.as_string())
    smtp_server.quit()

