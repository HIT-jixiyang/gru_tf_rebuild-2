import time
import numpy as np
import pandas as pd
import logging
import smtplib
from email.mime.text import MIMEText
from email.header import Header
ERROR_MAP={1:'MISS FRAME OR ERROR FRAM',2:'balabala'}
def auto_create_same_file(src_dir,des_dir,start,periods):
    dates = pd.date_range(start=start, periods=periods, freq='6T')
    file_path='/extend/deploy_test/18_2500_radar/cappi_ref_201803192000_2500_0.ref'
    file_name='cappi_ref_201803192000_2500_0.ref'

    for i in range(len(dates)):
        date=dates[i].strftime('%Y%m%d%H%M')
        src_file_name=src_dir+'/'+file_name.replace('201803192000',date)
        des_file_name=des_dir+'/'+file_name.replace('201803192000',date)
        ref = np.fromfile(src_file_name, dtype=np.uint8)
        ref.tofile(des_file_name)
        print('Auto create new file: ',des_file_name)
        time.sleep(60)
def auto_mail(error_code,msg):
    print('one mail will be sended')
    smtp=smtplib.SMTP()
    smtp.connect('smtp.qq.com')
    smtp.login('1044456468@qq.com','egtmjigyqmtcbcch')
    if ERROR_MAP[error_code] is not None:
        error_text=ERROR_MAP[error_code]+'-----'+msg
    else:
        error_text='未知原因'
    message = MIMEText('系统出错啦，赶快来看看啊！'
                       +'错误码:'+str(error_code)+'错误原因：'+error_text, 'plain', 'utf-8')
    message['From'] = Header("天气预报系统", 'utf-8')  # 发送者
    message['To'] = Header("17863136173@163.com", 'utf-8')  # 接收者
    subject = 'Python SMTP 邮件测试'
    message['Subject'] = Header(subject, 'utf-8')
    smtp.sendmail('1044456468@qq.com','17863136173@163.com',message.as_string())
    smtp.quit()
    print('end')
    pass

if __name__ == '__main__':
    auto_create_same_file('/extend/deploy_test/18_2500_radar','/tmp/refs','201803192300',10)
    # auto_mail(110,'error')