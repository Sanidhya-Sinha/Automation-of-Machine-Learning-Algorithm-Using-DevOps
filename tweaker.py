{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import keras\n",
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Dropout, Activation, Flatten\n",
    "from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D\n",
    "from keras.layers.normalization import BatchNormalization\n",
    "from keras.regularizers import l2\n",
    "from keras.datasets import mnist\n",
    "from keras.utils import np_utils as npu\n",
    "from keras.backend import clear_session"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Model \n",
    "(train_X , train_y), (test_X , test_y) = mnist.load_data(\"mymnist.data\")\n",
    "# Reshape data and change type\n",
    "test_X = test_X.reshape(-1 , 28, 28, 1)\n",
    "train_X = train_X.reshape(-1 ,  28, 28, 1)\n",
    "test_X = test_X.astype(\"float32\")\n",
    "train_X = train_X.astype(\"float32\")\n",
    "# One hot encoding \n",
    "test_y = npu.to_categorical(test_y)\n",
    "train_y = npu.to_categorical(train_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28, 1)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "********TRIAL :  1\n",
      "Train on 60000 samples, validate on 10000 samples\n",
      "Epoch 1/1\n",
      "60000/60000 [==============================] - 10s 159us/step - loss: 3.0691 - accuracy: 0.1111 - val_loss: 2.2990 - val_accuracy: 0.1150\n",
      "Test loss : 229.90449825286868\n",
      "-------Accuracy of the model : 11.500000208616257\n",
      "_______________________________________________________\n",
      "\n",
      "\n",
      "********TRIAL :  3\n",
      "Train on 60000 samples, validate on 10000 samples\n",
      "Epoch 1/2\n",
      "60000/60000 [==============================] - 13s 222us/step - loss: 1.7574 - accuracy: 0.5380 - val_loss: 0.8965 - val_accuracy: 0.7265\n",
      "Epoch 2/2\n",
      "60000/60000 [==============================] - 13s 215us/step - loss: 0.7388 - accuracy: 0.7732 - val_loss: 0.5004 - val_accuracy: 0.8931\n",
      "Test loss : 50.04211278915405\n",
      "-------Accuracy of the model : 89.31000232696533\n",
      "_______________________________________________________\n",
      "\n",
      "\n",
      "********TRIAL :  5\n",
      "Train on 60000 samples, validate on 10000 samples\n",
      "Epoch 1/3\n",
      "60000/60000 [==============================] - 20s 335us/step - loss: 0.7692 - accuracy: 0.8077 - val_loss: 0.1786 - val_accuracy: 0.9520\n",
      "Epoch 2/3\n",
      "60000/60000 [==============================] - 20s 336us/step - loss: 0.1303 - accuracy: 0.9661 - val_loss: 0.1094 - val_accuracy: 0.9704\n",
      "Epoch 3/3\n",
      "60000/60000 [==============================] - 20s 340us/step - loss: 0.0755 - accuracy: 0.9785 - val_loss: 0.1022 - val_accuracy: 0.9744\n",
      "Test loss : 10.223161492389627\n",
      "-------Accuracy of the model : 97.4399983882904\n",
      "_______________________________________________________\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "accuracy= open(\"accuracy.txt\",\"r\")\n",
    "accuracy = float(accuracy.read())\n",
    "\n",
    "#Initials\n",
    "neurons = 10\n",
    "epochs = 1\n",
    "test = 1\n",
    "flag = 0\n",
    "kernel = 8\n",
    "batch_size = 128\n",
    "#filter = 3\n",
    "\n",
    "\n",
    "while int(accuracy)<90:\n",
    "    if flag == 1 :\n",
    "        model = keras.backend.clear_session()\n",
    "        neurons = neurons+10\n",
    "        epochs = epochs+1 \n",
    "        test = test + 1\n",
    "        kernel = kernel * 2\n",
    "        test = test + 1\n",
    "    print(\"* * * TRIAL : \",test ,\"-----------------\")\n",
    "    model=Sequential()    \n",
    "    model.add(Conv2D(kernel, (3,3), input_shape = (28, 28, 1), activation = 'relu'))\n",
    "    model.add(MaxPooling2D(pool_size =(2,2)))\n",
    "    model.add(Flatten())\n",
    "    model.add(Dense(neurons, activation = 'relu'))\n",
    "    model.add(Dense(10, activation = 'softmax'))\n",
    "    model.compile( optimizer= \"Adam\" , loss='categorical_crossentropy',  metrics=['accuracy'] )\n",
    "    train_X.shape\n",
    "    model_predict= model.fit(train_X, train_y,batch_size=batch_size,verbose=1,epochs=epochs,validation_data=(test_X, test_y),shuffle=True)\n",
    "    scores = model.evaluate(test_X, test_y, verbose=False)\n",
    "    print('Test loss :', scores[0]*100)\n",
    "    print('-------Accuracy of the model :', scores[1]*100)\n",
    "    accuracy = scores[1]*100\n",
    "    print(\"_______________________________________________________\")\n",
    "    print()\n",
    "    print()\n",
    "    flag = 1\n",
    "    \n",
    "print(\"Total numbers of epochs :\" , epochs)\n",
    "print(\"Total number of filters :\", kernel)\n",
    "print(\"Total number of neurons :\", neurons)\n",
    "print(\"Final Accuracy : \", accuracy)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import smtplib\n",
    "s = smtplib.SMTP('smtp.gmail.com', 587)\n",
    "s.starttls()\n",
    "\n",
    "s.login(\"sanidhyasinha2000@gmail.com\", \"welcomedam\")\n",
    "\n",
    "\n",
    "    # message\n",
    "message_success = \"Achieved your desired accuracy . Congrats :)\"\n",
    "    \n",
    "\n",
    "    # sending the mail \n",
    "s.sendmail(\"sanidhyasinha2000@gmail.com\", \"1706355@kiit.ac.in\", message_success)\n",
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
