import smtplib
import email.utils
from email.mime.text import MIMEText

# 编写邮件内容
message = MIMEText("content")

message['To'] = email.utils.formataddr(('sun', 'sunweihao97@gmail.com'))
message['From'] = email.utils.formataddr(('weihao', '2309598788@qq.com'))
message['Subject'] = 'subject Test'

# 登录服务器并发送
server = smtplib.SMTP_SSL('smtp.qq.com', 465)
server.login('2309598788@qq.com', 'aoyqtzjhzmxaeafg')  # 替换为你的 QQ 邮箱 SMTP 授权码

server.set_debuglevel(True)

try:
    server.sendmail('2309598788@qq.com', ['sunweihao97@gmail.com'], msg=message.as_string())
finally:
    server.quit()