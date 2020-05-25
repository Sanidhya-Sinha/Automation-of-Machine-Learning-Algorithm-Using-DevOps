{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
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
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28)"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(x_train, y_train), (x_test, y_test)  = mnist.load_data()\n",
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "size of image (28, 28)\n"
     ]
    }
   ],
   "source": [
    "img = x_train[0].shape\n",
    "print(\"size of image\",img)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = x_train.reshape(-1, 28, 28, 1)\n",
    "x_test = x_test.reshape(-1, 28,28 , 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28, 1)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = x_train.astype('float32')\n",
    "x_test = x_test.astype('float32')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train = npu.to_categorical(y_train)\n",
    "y_test = npu.to_categorical(y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "kernel = 8\n",
    "filter = 3\n",
    "pool = 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Conv2D(kernel, (filter,filter), input_shape = (28, 28, 1), activation = 'relu'))\n",
    "model.add(MaxPooling2D(pool_size =(pool,pool)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Flatten())\n",
    "\n",
    "model.add(Dense(10, activation = 'relu'))\n",
    "#model.add(Dropout(0.5))\n",
    "model.add(Dense(10, activation = 'softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "#model.add(Dense(units = 20 , input_dim = 28*28 , activation = 'relu'))\n",
    "#model.add(Dense(units=60  , input_dim = 28*28 , activation = 'relu'))\n",
    "#model.add(Dense(units=10 , input_dim = 28*28 , activation = 'softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile( optimizer= \"Adam\" , loss='categorical_crossentropy',  metrics=['accuracy'] )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_1 (Conv2D)            (None, 26, 26, 8)         80        \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 13, 13, 8)         0         \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 1352)              0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 10)                13530     \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 10)                110       \n",
      "=================================================================\n",
      "Total params: 13,720\n",
      "Trainable params: 13,720\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(model.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 60000 samples, validate on 10000 samples\n",
      "Epoch 1/1\n",
      "60000/60000 [==============================] - 18s 296us/step - loss: 2.4971 - accuracy: 0.1937 - val_loss: 1.9278 - val_accuracy: 0.2651\n",
      "Test loss: 1.927788631439209\n",
      "Test accuracy: 0.26510000228881836\n"
     ]
    }
   ],
   "source": [
    "batch_size = 128\n",
    "epochs = 1\n",
    "\n",
    "history = model.fit(x_train, y_train,\n",
    "          batch_size=batch_size,verbose=1,\n",
    "          epochs=epochs,\n",
    "          validation_data=(x_test, y_test),\n",
    "          shuffle=True)\n",
    "\n",
    "model.save(\"mnist_LeNet.h5\")\n",
    "\n",
    "# Evaluate the performance of our trained model\n",
    "scores = model.evaluate(x_test, y_test, verbose=False)\n",
    "print('Test loss:', scores[0])\n",
    "print('Test accuracy:', scores[1])\n",
    "accuracy = scores[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy is :  0.26510000228881836 %\n"
     ]
    }
   ],
   "source": [
    "f= open(\"accuracy.txt\",\"w+\")\n",
    "f.write(str(accuracy))\n",
    "f.close()\n",
    "print(\"Accuracy is : \" , accuracy ,\"%\")"
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
