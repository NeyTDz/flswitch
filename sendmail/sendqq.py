import smtplib
import sys
from email.mime.text import MIMEText
from email.header import Header
import yagmail

sys.path.append('..')
from train_params import *


def sendsmt(Info=""):
    # 第三方 SMTP 服务
    mail_host="smtp.qq.com"  #设置服务器
    mail_user="hunterzolomon@qq.com"    #用户名
    mail_pass="zzkhfrpuakeuhcjh"   #口令 
    
    sender = 'hunterzolomon@qq.com'
    receivers = ['hunterzolomon@qq.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
    
    message = MIMEText("Top:{}\nSparse:{}\nPower_choice:{}\nControl accuracy:{}\nEpochs:{}\n".format(K,SPARSE,POWER_CHOICE,CON_ACC,EPOCH), 'plain', 'utf-8')
    message['From'] = Header(MAIL_FROM, 'utf-8')
    message['To'] =  Header("", 'utf-8')
    
    subject = Info+" "+MAIL_TITLE
    message['Subject'] = Header(subject, 'utf-8')
    
    try:
        smtpObj = smtplib.SMTP() 
        smtpObj.connect(mail_host, 25)    # 25 为 SMTP 端口号
        smtpObj.login(mail_user,mail_pass)  
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("Message successfully send")
    except smtplib.SMTPException as e:
        print("Message not send")

def sendyagmail(Info=''):
    mail_host="smtp.qq.com"  #设置服务器
    mail_user="hunterzolomon@qq.com"    #用户名
    mail_pass="zzkhfrpuakeuhcjh"   #口令 
    receiver = 'hunterzolomon@qq.com'
    subject = Info+" "+MAIL_TITLE
    text = "Top:{}\nSparse:{}\nPower_choice:{}\nControl accuracy:{}\nEpochs:{}\n".format(K,SPARSE,POWER_CHOICE,CON_ACC,EPOCH)

    yag=yagmail.SMTP(user=mail_user,password=mail_pass,host=mail_host) #port=25,smtp_ssl=False，默认是ssl模式，如果要设置为false，则端口要是用25
    contents=[text] # ttt发送附件，也可以发送html格式
    yag.send(receiver,subject,contents=contents) 
    print("Message successfully send")


if __name__ == "__main__":
    sendyagmail()