import smtplib
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()

s.login("sanidhyasinha2000@gmail.com", "**********")


    # message
message_success = "Achieved your desired accuracy without tweeking . Congrats "


    # sending the mail
s.sendmail("sanidhyasinha2000@gmail.com", "1706355@kiit.ac.in", message_success)


    # terminating the session
s.quit()
