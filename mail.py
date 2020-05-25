{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "SMTPAuthenticationError",
     "evalue": "(535, b'5.7.8 Username and Password not accepted. Learn more at\\n5.7.8  https://support.google.com/mail/?p=BadCredentials g27sm12431915pfr.51 - gsmtp')",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mSMTPAuthenticationError\u001b[0m                   Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-4a90be736ff0>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0ms\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstarttls\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m \u001b[0ms\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlogin\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"sanidhyasinha2000@gmail.com\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"welcomedam\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      6\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      7\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\envs\\mykeras\\lib\\smtplib.py\u001b[0m in \u001b[0;36mlogin\u001b[1;34m(self, user, password, initial_response_ok)\u001b[0m\n\u001b[0;32m    728\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    729\u001b[0m         \u001b[1;31m# We could not login successfully.  Return result of last attempt.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 730\u001b[1;33m         \u001b[1;32mraise\u001b[0m \u001b[0mlast_exception\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    731\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    732\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mstarttls\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkeyfile\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcertfile\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcontext\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\envs\\mykeras\\lib\\smtplib.py\u001b[0m in \u001b[0;36mlogin\u001b[1;34m(self, user, password, initial_response_ok)\u001b[0m\n\u001b[0;32m    719\u001b[0m                 (code, resp) = self.auth(\n\u001b[0;32m    720\u001b[0m                     \u001b[0mauthmethod\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mgetattr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmethod_name\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 721\u001b[1;33m                     initial_response_ok=initial_response_ok)\n\u001b[0m\u001b[0;32m    722\u001b[0m                 \u001b[1;31m# 235 == 'Authentication successful'\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    723\u001b[0m                 \u001b[1;31m# 503 == 'Error: already authenticated'\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\envs\\mykeras\\lib\\smtplib.py\u001b[0m in \u001b[0;36mauth\u001b[1;34m(self, mechanism, authobject, initial_response_ok)\u001b[0m\n\u001b[0;32m    640\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mcode\u001b[0m \u001b[1;32min\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m235\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m503\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    641\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mcode\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mresp\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 642\u001b[1;33m         \u001b[1;32mraise\u001b[0m \u001b[0mSMTPAuthenticationError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcode\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mresp\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    643\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    644\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mauth_cram_md5\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mchallenge\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mSMTPAuthenticationError\u001b[0m: (535, b'5.7.8 Username and Password not accepted. Learn more at\\n5.7.8  https://support.google.com/mail/?p=BadCredentials g27sm12431915pfr.51 - gsmtp')"
     ]
    }
   ],
   "source": [
    "import smtplib\n",
    "s = smtplib.SMTP('smtp.gmail.com', 587)\n",
    "s.starttls()\n",
    "\n",
    "s.login(\"sanidhyasinha2000@gmail.com\", \"welcomedam\")\n",
    "\n",
    "\n",
    "    # message\n",
    "message = \"Your accuracy is less than 90% :(\"\n",
    "    \n",
    "\n",
    "    # sending the mail \n",
    "s.sendmail(\"sanidhyasinha2000@gmail.com\", \"1706355@kiit.ac.in\", message)\n",
    "    \n",
    "\n",
    "    # terminating the session \n",
    "s.quit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
