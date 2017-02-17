INSTALLATION OF NN PROJECT
==========================

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
