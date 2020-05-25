{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: '/mlops/program.py'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-93953120bdb5>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mprogramfile\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'/mlops/program.py'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;34m'r'\u001b[0m\u001b[1;33m)\u001b[0m     \u001b[1;31m#connecting to the code file\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mcode\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mprogramfile\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mread\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m                               \u001b[1;31m#reading the code file\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;32mif\u001b[0m \u001b[1;34m'keras'\u001b[0m \u001b[1;32mor\u001b[0m \u001b[1;34m'tensorflow'\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mcode\u001b[0m\u001b[1;33m:\u001b[0m                     \u001b[1;31m#because keras or tensorflow keyword is a must for a deep learning program\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[1;34m'Conv2D'\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mcode\u001b[0m\u001b[1;33m:\u001b[0m                            \u001b[1;31m#beacuse if a code is of CNN conv2D layer is a must\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: '/mlops/program.py'"
     ]
    }
   ],
   "source": [
    "programfile = open('/mlops/program.py','r')\t#connecting to the code file\n",
    "code = programfile.read()\t\t\t\t#reading the code file\n",
    "\n",
    "if 'keras' or 'tensorflow' in code:\t\t\t#because keras or tensorflow keyword is a must for a deep learning program\n",
    "\tif 'Conv2D' in code:\t\t\t\t#beacuse if a code is of CNN conv2D layer is a must \n",
    "\t\tprint('Present')\n",
    "\telse:\n",
    "\t\tprint('Not a CNN')\n",
    "else:\n",
    "\tprint('NOT A NEURAL NETWORK')"
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
