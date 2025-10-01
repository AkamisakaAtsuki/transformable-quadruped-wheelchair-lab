import itertools
import os
import toml

from setuptools import setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = [
    # "numpy",
    # "torch==2.4.0",
    # "torchvision>=0.14.1",  # ensure compatibility with torch 1.13.1
    # "protobuf>=3.20.2, < 5.0.0",
    # "h5py",
    # "tensorboard",
    # "moviepy",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

current_dir = os.path.dirname(os.path.abspath(__file__))
rsl_rl_path = os.path.join(current_dir, "..", "rsl_rl")

EXTRAS_REQUIRE = {
    "rsl-rl": [rsl_rl_path],
}

EXTRAS_REQUIRE["rsl_rl"] = EXTRAS_REQUIRE["rsl-rl"]

EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))

EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))

setup(
    name="transformable_quadruped_wheelchair_isaaclab",
    packages=["transformable_quadruped_wheelchair_isaaclab"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    # license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    # classifiers=[
    #     "Natural Language :: English",
    #     "Programming Language :: Python :: 3.10",
    #     "Isaac Sim :: 4.2.0",
    # ],
    zip_safe=False,
)