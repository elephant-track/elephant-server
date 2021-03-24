[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elephant-track/elephant-server/blob/main/elephant_server.ipynb)

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

### ELEPHANT Server Requirements (Docker)

|                  | Requirements                                                                                                                                                                                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Operating System | Linux-based OS compatible with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)                                                                                                                                             |
| Docker           | [Docker](https://www.docker.com/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (see [supported versions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#container-runtimes)) |
| GPU              | NVIDIA CUDA GPU with sufficient VRAM for your data (recommended: 11 GB or higher)                                                                                                                                                                                                           |
| Storage          | Sufficient size for your data (recommended: 1 TB or higher)                                                                                                                                                                                                                                 |
### ELEPHANT Server Requirements (Singularity)

|                  | Requirements                                                                                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Operating System | Linux-based OS                                                                                                                                                  |
| Singularity      | [Singularity](https://sylabs.io/guides/3.7/user-guide/index.html) (see [requirements for NVIDIA GPUs & CUDA](https://sylabs.io/guides/3.7/user-guide/gpu.html)) |
| GPU              | NVIDIA CUDA GPU with sufficient VRAM for your data (recommended: 11 GB or higher)                                                                               |
| Storage          | Sufficient size for your data (recommended: 1 TB or higher)                                                                                                     |

### Setting up the ELEPHANT Server

There are three options to set up the ELEPHANT server.

- <a href="#/?id=setting-up-with-docker" onclick="alwaysScroll(event)">Setting up with Docker</a>
  
  This option is recommended if you have a powerful computer that satisfies <a href="#/?id=elephant-server-requirements-docker" onclick="alwaysScroll(event)">the server requirements (Docker)</a> with root privileges.

- <a href="#/?id=setting-up-with-singularity" onclick="alwaysScroll(event)">Setting up with Singularity</a>
  
  This option is recommended if you can access a powerful computer that satisfies <a href="#/?id=elephant-server-requirements-singularity" onclick="alwaysScroll(event)">the server requirements (Singularity)</a> as a non-root user (e.g. HPC cluster).

- <a href="#/?id=setting-up-with-google-colab" onclick="alwaysScroll(event)">Setting up with Google Colab</a>
  
  Alternatively, you can set up the ELEPHANT server with [Google Colab](https://research.google.com/colaboratory/faq.html), a freely available product from Google Research. In this option, you don't need to have a high-end GPU or a Linux machine to start using ELEPHANT's deep learning capabilities.

#### Setting up with Docker

##### Prerequisite

Please check that your computer meets <a href="#/?id=elephant-server-requirements" onclick="alwaysScroll(event)">the server requirements</a>.

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

##### 1.Download/Clone a repository

Download and extract a [.zip file](https://github.com/elephant-track/elephant-server/releases/download/v0.1.0/elephant-server-0.1.0.zip).

Alternatively, you can clone a repository from [GitHub](https://github.com/elephant-track/elephant-server).

```bash
git clone https://github.com/elephant-track/elephant-server.git
```

##### 2. Build a Docker image

First, change the directory to the project root.

```bash
cd elephant-server-0.1.0
```

The following command will build a Docker image that integrates all the required modules.

```bash
make build
```

##### 3. Generate a dataset for the ELEPHANT server

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
export ELEPHANT_WORKSPACE="YOUR_WORKSPACE_DIR"
make bash
```

```bash
# This is optional
make ELEPHANT_WORKSPACE="YOUR_WORKSPACE_DIR" bash
```

| Info <br> :information_source: | Multi-view data is not supported by ELEPHANT. You need to create a fused data (e.g. with [BigStitcher Fuse](https://imagej.net/BigStitcher_Fuse)) before converting to `.zarr` . |
| :----------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

##### 4. Launch the ELEPHANT server via Docker

The ELEPHANT server is accompanied by several services, including [Flask](https://flask.palletsprojects.com/en/1.1.x/),
[uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/), [NGINX](https://www.nginx.com/), [redis](https://redis.io/)
and [RabbitMQ](https://www.rabbitmq.com/).
These services are organized by [Supervisord](http://supervisord.org/) inside the Docker container,
exposing the port `8080` for [NGINX](https://www.nginx.com/) and `5672` for [RabbitMQ](https://www.rabbitmq.com/) available on `localhost`. 

```bash
make launch # launch the services
```

Now, the ELEPHANT server is ready.

#### Setting up with Singularity

##### Prerequisite

`Singularity >= 3.6.0` is required. Please check the version of Singularity on your system.

```bash
singularity --version
```

##### 1. Build a container

Run the following command at the project root directory where you can find a `elephant.def` file.

```bash
singularity build --fakeroot elephant.sif elephant.def
```

##### 2. Prepare files to bind

The following command copies `/var/lib/`, `/var/log/` and `/var/run/` in the container to `$HOME/.elephant_binds` on the host.

```bash
singularity run --fakeroot elephant.sif
```

##### 3. Start an instance for the ELEPHANT server

It is recommended to launch the ELEPHANT server inside a singularity `instance` rather than using `shell` or `exec` directly, which can make some processes alive after exiting the `supervisor` process. All processes inside a `instance` can be terminated by stopping the `instance` ([see details](https://sylabs.io/guides/3.7/user-guide/running_services.html#container-instances-in-singularity)).

The command below starts an `instance` named `elephant` using the image written in `elephant.sif`.\
The `--nv` option is required to set up the container that can use NVIDIA GPU and CUDA ([see details](https://sylabs.io/guides/3.7/user-guide/gpu.html)).\
The `--bind` option specifies the directories to bind from the host to the container ([see details](https://sylabs.io/guides/3.7/user-guide/bind_paths_and_mounts.html)). The files copied in the previous step are bound to the original container location as `writable` files. Please set `$ELEPHANT_WORKSPACE` to the `workspace` directory on your system.

```bash
singularity instance start --nv --bind $HOME/.elephant_binds/var/lib:/var/lib,$HOME/.elephant_binds/var/log:/var/log,$HOME/.elephant_binds/var/run:/var/run,$ELEPHANT_WORKSPACE:/workspace elephant.sif elephant
```

##### 4. Generate a dataset for the ELEPHANT server

The following command will generate a dataset for the ELEPHANT server.
Please see details in <a href="#/?id=_3-generate-a-dataset-for-the-elephant-server" onclick="alwaysScroll(event)">the Docker part</a>.

```bash
singularity exec instance://elephant python /opt/elephant/script/dataset_generator.py --uint16 /workspace/datasets/elephant-demo/elephant-demo.h5 /workspace/datasets/elephant-demo
```

##### 5. Launch the ELEPHANT server

The following command execute a script that launches the ELEPHANT server.
Please specify the `SINGULARITYENV_CUDA_VISIBLE_DEVICES` if you want to use a specific GPU device on your system (default: `0`).

```bash
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec instance://elephant /start.sh
```

At this point, you will be able to work with the ELEPHANT server.
Please follow <a href="#/?id=remote-connection-to-the-elephant-server" onclick="alwaysScroll(event)">the instructions for seting up the remote connection</a>.

##### 6. Stop an instance for the ELEPHANT server

After exiting the `exec` by `Ctrl+C`, please do not forget to stop the `instance`.

```bash
singularity instance stop elephant
```

#### Setting up with Google Colab

##### 1. Prepare a Google account

If you already have one, you can just use it. Otherwise, create a Google account [here](https://accounts.google.com/signup).

##### 2. Create a ngrok account

Create a ngrok account from the following link.

[ngrok - secure introspectable tunnels to localhost](https://dashboard.ngrok.com/signup)

##### 3. Open and run a Colab notebook

Open a Colab notebook from this button. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elephant-track/elephant-server/blob/main/elephant_server.ipynb)

On Goolge Colab, run the command [Runtime > Run all] and select `RUN ANYWAY` in the following box.

<img src="../assets/colab-warning.png"></img>

##### 4. Start a ngrok tunnel

After around 10 minutes, you will find the following box on the bottom of the page.

<img src="../assets/ngrok-box.png"></img>

Click the link to open your ngrok account page and copy your authtoken, then paste it to the box above.

<img src="../assets/ngrok-authtoken.png"></img>

After inputting your authtoken, you will have many lines of outputs. Scroll up and find the following two lines.

```Colab
SSH command: ssh -p[your_random_5digits] root@[your_random_value].tcp.ngrok.io
Root password: [your_random_password]
```

##### 5. Establish connections from your computer to the server on Colab

On your computer, launch a powershell (Windows) or terminal (Mac&Linux) and run the following command. Please leave the powershell/terminal window open.

| Info <br> :information_source: | Please do not forget to replace `your_random_5digits` and `your_random value`. When you are asked a password, use the `your_random_password` found in the previous step. |
| :----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

Windows:

```Powershell
ssh.exe -N -L 8080:localhost:80 -o PubkeyAuthentication=no -o TCPKeepAlive=yes -o ServerAliveInterval=30 -p[your_random_5digits] root@[your_random value].tcp.ngrok.io
```

Mac&Linux:

```bash
ssh -N -L 8080:localhost:80 -o PubkeyAuthentication=no -o TCPKeepAlive=yes -o ServerAliveInterval=30 -p[your_random_5digits] root@[your_random value].tcp.ngrok.io
```

Continue with `yes` if you are asked the following question.

```
Are you sure you want to continue connecting (yes/no)? 
```

Launch another powershell (Windows) or terminal (Mac&Linux) and run the following command. Please leave the powershell/terminal window open.

Windows:

```Powershell
ssh.exe -N -L 5672:localhost:5672 -o PubkeyAuthentication=no -o TCPKeepAlive=yes -o ServerAliveInterval=30 -p[your_random_5digits] root@[your_random value].tcp.ngrok.io
```

Mac&Linux:

```
ssh -N -L 5672:localhost:5672 -o PubkeyAuthentication=no -o TCPKeepAlive=yes -o ServerAliveInterval=30 -p[your_random_5digits] root@[your_random value].tcp.ngrok.io
```

##### 6. Terminate

When you finish using the ELEPHANT, stop and terminate your Colab runtime so that you can release your resources.

- Stop the running execution by [Runtime > Interrupt execution]
- Terminate the runtime by [Runtime > Manage sessions]

<img src="../assets/terminate-colab.png"></img>

| Info <br> :information_source: | If you see the following message, it is likely that you exceeded the usage limits. Unfortunately, you cannot use Colab with GPU at the moment. See details <a href="https://research.google.com/colaboratory/faq.html#usage-limits">here</a> |
| :----------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

<img src="../assets/colab-limits-warning.png"></img>

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
- [ngrok](https://ngrok.com/)
- [Google Colab](https://colab.research.google.com)
  
## Citation

Please cite our paper.

## License

[BSD-2-Clause](LICENSE)