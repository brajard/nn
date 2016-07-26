Example of Neural Network on oceanography images
================================================

Useful link ?
-------------

* overview : https://martin-thoma.com/lasagne-for-python-newbies/
* auto-encoder using lasagne overview : https://swarbrickjones.wordpress.com/2015/04/29/convolutional-autoencoders-in-pythontheanolasagne/
* example code of auto-encoder : https://github.com/mikesj-public/convolutional_autoencoder/blob/master/mnist_conv_autoencode.py
* toolbox keras for designing recurrent nn : http://keras.io/

Install
-------
`pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt`

`pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git`

`pip install --upgrade https://github.com/Theano/Theano/archive/master.zip`

`pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip`

`pip install keras`

#`pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps`

Tutorial
--------
* To run an example of CNN, see  [lasagne/script_test.py](lasagne/script_test.py)

* An example of a simple RNN, see [lasagne/recurrent_test.py](lasagne/recurrent_test.py)

* An example of CNN + RNN, see  [lasagne/combo_test.py](lasagne/combo_test.py)