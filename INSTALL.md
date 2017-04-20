INSTALLATION OF NN PROJECT
==========================

Installing Anaconda
-------------------
After downloading the Anaconda installer, run the following command from a terminal:
```Shell
$ bash Anaconda-2.x.x-Linux-x86[_64].sh
```
You then need to edit your .bashrc file to include the ECHO $PATH of where you installed Anaconda.
You normally get notified about this at the end of the instalation.

Using Conda
-----------
* `conda create --name nn python=3`
* `source activate nn`
* (opt) `pip install ipython`
* `pip install scipy`
* `pip install numpy`
* `pip install pandas`
* `pip install xarray`
* `pip install matplotlib`
* `pip install --upgrade https://github.com/Theano/Theano/archive/master.zip`
* `pip install keras`
* `pip install sklearn`
* `pip install h5py`
* `conda install mkl-service`
* for visualization :
  * `pip install pydot-ng`
  * `conda install graphviz`
* Modify your `$HOME/.theanorc` file:
```Shell
[global]
floatX = float32
device = cpu

[nvcc]
fastmath = True
```
* Modify your `~/.keras/keras.json` file:
```JSON
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "theano",
    "image_dim_ordering": "th"
}
```
* Test if it works : `./script_train_test.py`

Using GPU
---------
* Modify your `$HOME/.theanorc` file:

* Modify your `$HOME/.bashrc`file to add the path and library path of cuda :
e.g
```
export CPATH="/usr/local/cuda/include:$CPATH"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

