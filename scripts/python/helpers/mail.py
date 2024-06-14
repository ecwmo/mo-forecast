import os
import yagmail
import json
from typing import List

from config import Config


def send_mail(msg: str, to: List[str] = None, subject: str = None):
    conf = Config()
    conf_file = open(conf.script_dir / "python/config/mail.json")
    mail_conf = json.load(conf_file)
    conf_file.close()

    username = os.getenv("MAIL_USERNAME")
    if username is None:
        raise ValueError("MAIL_USERNAME")

    password = os.getenv("MAIL_PASSWORD")
    if password is None:
        raise ValueError("MAIL_PASSWORD")

    mail = yagmail.SMTP(username, password)

    if to is None:
        to = mail_conf.get("to", [])
    if subject is None:
        subject = mail_conf.get("subject", "[Alert]")
    if (len(to) == 0) or (len(msg) == 0):
        return
    if len(to) > 1:
        mail.send(to=to[0], bcc=to[1:], subject=subject, contents=msg)
    else:
        mail.send(to=to, subject=subject, contents=msg)
