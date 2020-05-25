import smtplib
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()

s.login("sanidhyasinha2000@gmail.com", "welcomedam")


    # message
message = "Your accuracy is less than 90% .Try again"


    # sending the mail
s.sendmail("sanidhyasinha2000@gmail.com", "1706355@kiit.ac.in", message)


    # terminating the session
s.quit()
