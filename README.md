## ELEPHANT: Tracking cell lineages in 3D by incremental deep learning

<table>
  <tbody>
    <tr>
      <th rowspan=7><img src="../assets/incremental-training-demo.gif?raw=true"></img></th>
    </tr>
    <tr>
      <th colspan=2><img src="../assets/elephant-logo-text.svg" height="64px"></th>
    </tr>
    <tr>
      <td>Developer</td>
      <td><a href="http://www.ens-lyon.fr/lecole/nous-connaitre/annuaire/ko-sugawara">Ko Sugawara</a></td>
    </tr>
    <tr>
      <td valign="top">Forum</td>
      <td><a href="https://forum.image.sc/tag/elephant">Image.sc forum</a><br>Please post feedback and questions to the forum.<br>It is important to add the tag <code>elephant</code> to your posts so that we can reach you quickly.</td>
    </tr>
    <tr>
      <td>Source code</td>
      <td><a href="https://github.com/elephant-track">GitHub</a></td>
    </tr>
    <tr>
      <td>Publication</td>
      <td><a href="https://www.biorxiv.org/content">bioRxiv</a></td>
    </tr>
  </tbody>
</table>


---
ELEPHANT is a platform for 3D cell tracking, based on incremental and interactive deep learning.

It works on client-server architecture. The server is built as a web application that serves deep learning-based algorithms.

This repository provides an implementation of the ELEPHANT server. The ELEPHANT client can be found [here](https://github.com/elephant-track/elephant-client).

Please refer to [the documentation]() for details.

---

### ELEPHANT Server Requirements

|                  | Requirements                                                                                                                                                                                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Operating System | Linux-based OS compatible with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)                                                                                                                                             |
| Docker           | [Docker](https://www.docker.com/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (see [supported versions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#container-runtimes)) |
| GPU              | NVIDIA CUDA GPU with sufficient VRAM for your data (recommended: 11 GB or higher)                                                                                                                                                                                                           |
| Storage          | Sufficient size for your data (recommended: 1 TB or higher)                                                                                                                                                                                                                                 |

| Info <br> :information_source: | The total amount of data can be 10-30 times larger than the original data size when the prediction outputs (optional) are generated. |
| :----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------- |

### Setting up the ELEPHANT Server

#### Prerequisite

Install [Docker](https://www.docker.com/) with [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

By defaut, ELEPHANT assumes you can [run Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/).\
If you need to run `Docker` with `sudo`, please set the environment variable `ELEPHANT_DOCKER` as below.

```bash
export ELEPHANT_DOCKER="sudo docker"
```

Alternatively, you can set it at runtime.

```bash
make ELEPHANT_DOCKER="sudo docker" bash
```

#### 1.Download/Clone a repository

Download and extract a [.zip file](https://github.com/elephant-track/elephant-server/releases/download/v0.1.0/elephant-server-0.1.0.zip).

Alternatively, you can clone a repository from [GitHub](https://github.com/elephant-track/elephant-server).

```bash
git clone https://github.com/elephant-track/elephant-server.git
```

#### 2. Build a Docker image

First, change the directory to the project root.

```bash
cd elephant-server-0.1.0
```

The following command will build a Docker image that integrates all the required modules.

```bash
make build
```

#### 3. Generate a dataset for the ELEPHANT server

Please [prepare](https://imagej.net/BigDataViewer.html#Exporting_from_ImageJ_Stacks) your image data, producing a pair of [BigDataViewer](https://imagej.net/BigDataViewer) `.h5` and `.xml` files, or [download the demo data](https://doi.org/10.5281/zenodo.4549193) and extract it as below.

The ELEPHANT server deals with images using [Zarr](https://zarr.readthedocs.io/en/stable/). The following command generates required `zarr` files from the [BigDataViewer](https://imagej.net/BigDataViewer) `.h5` file.


```bash
workspace
├── datasets
│   └── elephant-demo
│       ├── elephant-demo.h5
│       └── elephant-demo.xml
```

Run the script inside a Docker container.

```bash
make bash # run bash inside a docker container
```

```bash
python /opt/elephant/script/dataset_generator.py --uint16 /workspace/datasets/elephant-demo/elephant-demo.h5 /workspace/datasets/elephant-demo
# usage: dataset_generator.py [-h] [--uint16] [--divisor DIVISOR] input output

# positional arguments:
#   input              input .h5 file
#   output             output directory

# optional arguments:
#   -h, --help         show this help message and exit
#   --uint16           with this flag, the original image will be stored with
#                      uint16
#                      default: False (uint8)
#   --divisor DIVISOR  divide the original pixel values by this value (with
#                      uint8, the values should be scale-downed to 0-255)

exit # exit from a docker container
```

You will find the following results.

```
workspace
├── datasets
│   └── elephant-demo
│       ├── elephant-demo.h5
│       ├── elephant-demo.xml
│       ├── flow_hashes.zarr
│       ├── flow_labels.zarr
│       ├── flow_outputs.zarr
│       ├── imgs.zarr
│       ├── seg_labels_vis.zarr
│       ├── seg_labels.zarr
│       └── seg_outputs.zarr
```

| Info <br> :information_source: | By default, the docker container is launched with [volumes](https://docs.docker.com/storage/volumes/), mapping the local `workspace/` directory to the `/workspace/` directory in the container. <br> The local workspace directory can be set by the `ELEPHANT_WORKSPACE` environment variable (Default: `${PWD}/workspace`). |
| :----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

```bash
# This is optional
export ELEPHANT_WORKSPACE="YOUR_DATASET_DIR"
make bash
```

```bash
# This is optional
make ELEPHANT_WORKSPACE="YOUR_DATASET_DIR" bash
```

| Info <br> :information_source: | Multi-view data is not supported by ELEPHANT. You need to create a fused data (e.g. with [BigStitcher Fuse](https://imagej.net/BigStitcher_Fuse)) before converting to `.zarr` . |
| :----------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

#### 4. Launch the ELEPHANT server via Docker

The ELEPHANT server is accompanied by several services, including [Flask](https://flask.palletsprojects.com/en/1.1.x/),
[uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/), [NGINX](https://www.nginx.com/), [Redis](https://redis.io/)
and [RabbitMQ](https://www.rabbitmq.com/).
These services are organized by [Supervisord](http://supervisord.org/) inside the Docker container,
exposing the port `8080` for [NGINX](https://www.nginx.com/) and `5672` for [RabbitMQ](https://www.rabbitmq.com/) available on `localhost`. 

```bash
make launch # launch the services
```

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Numpy](https://numpy.org/)
- [Scipy](https://www.scipy.org/)
- [scikit-image](https://scikit-image.org/)
- [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- [uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/)
- [NGINX](https://www.nginx.com/)
- [Redis](https://redis.io/)
- [RabbitMQ](https://www.rabbitmq.com/)
- [Supervisord](http://supervisord.org/)
- [uwsgi-nginx-flask-docker](https://github.com/tiangolo/uwsgi-nginx-flask-docker)
  
## Citation

Please cite our paper.

## License

[BSD-2-Clause](LICENSE)