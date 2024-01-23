import pathlib as pl

BASE = pl.Path(__file__).parent.resolve()

DATA = BASE / "data"
SUBSETCIFAR = DATA / "cifar10"
DATA.mkdir(exist_ok=True)
SUBSETCIFAR.mkdir(exist_ok=True)


PLOT_FOLDER = BASE / "plots"
PLOT_FOLDER.mkdir(exist_ok=True)

DATA_FOLDER = BASE / "data"


PLOT_FOLDER.mkdir(exist_ok=True)


BASEMODEL = BASE / "base_model"
BASEMODEL.mkdir(exist_ok=True)

MNIST_BASE = BASEMODEL / "mnist"
CIFAR10_BASE = BASEMODEL / "cifar10"
MNIST_BASE.mkdir(exist_ok=True)
CIFAR10_BASE.mkdir(exist_ok=True)


CNN_MNIST = MNIST_BASE / "cnn"
FC_MNIST = MNIST_BASE / "fcrelu"
CNN_CIFAR10 = CIFAR10_BASE / "cnn"
FC_CIFAR10 = CIFAR10_BASE / "fcrelu"
CNN_MNIST.mkdir(exist_ok=True)
FC_MNIST.mkdir(exist_ok=True)
CNN_CIFAR10.mkdir(exist_ok=True)
FC_CIFAR10.mkdir(exist_ok=True)

PGDL2_CNN_MNIST = CNN_MNIST/ "pgdl2"
AA_CNN_MNIST = CNN_MNIST/ "autoattack"
PGDL2_FCRELU_MNIST = FC_MNIST/ "pgdl2"
AA_FCRELU_MNIST = FC_MNIST/ "autoattack"

PGDL2_CNN_MNIST.mkdir(exist_ok=True)
AA_CNN_MNIST.mkdir(exist_ok=True)
PGDL2_FCRELU_MNIST.mkdir(exist_ok=True)
AA_FCRELU_MNIST.mkdir(exist_ok=True)