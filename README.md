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

Please refer to [the documentation](https://elephant-track.github.io/#/v0.2/) for details.

---

### Setting up the ELEPHANT Server

There are three options to set up the ELEPHANT server.

- <a href="https://elephant-track.github.io/#/?id=setting-up-with-docker" onclick="alwaysScroll(event)">Setting up with Docker</a>
  
  This option is recommended if you have a powerful computer that satisfies <a href="https://elephant-track.github.io/#/?id=elephant-server-requirements-docker" onclick="alwaysScroll(event)">the server requirements (Docker)</a> with root privileges.

- <a href="https://elephant-track.github.io/#/?id=setting-up-with-singularity" onclick="alwaysScroll(event)">Setting up with Singularity</a>
  
  This option is recommended if you can access a powerful computer that satisfies <a href="https://elephant-track.github.io/#/?id=elephant-server-requirements-singularity" onclick="alwaysScroll(event)">the server requirements (Singularity)</a> as a non-root user (e.g. HPC cluster).

- <a href="https://elephant-track.github.io/#/?id=setting-up-with-google-colab" onclick="alwaysScroll(event)">Setting up with Google Colab</a>
  
  Alternatively, you can set up the ELEPHANT server with [Google Colab](https://research.google.com/colaboratory/faq.html), a freely available product from Google Research. In this option, you don't need to have a high-end GPU or a Linux machine to start using ELEPHANT's deep learning capabilities.

The detailed instructions for each option can be found in [the documentation](https://elephant-track.github.io/#/v0.2/).

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

Please cite our paper on [bioRxiv](https://biorxiv.org/cgi/content/short/2021.02.26.432552v1).

- Sugawara, K., Cevrim, C. & Averof, M. [*Tracking cell lineages in 3D by incremental deep learning.*](https://biorxiv.org/cgi/content/short/2021.02.26.432552v1) bioRxiv 2021. doi:10.1101/2021.02.26.432552

```.bib
@article {Sugawara2021.02.26.432552,
	author = {Sugawara, Ko and Cevrim, Cagri and Averof, Michalis},
	title = {Tracking cell lineages in 3D by incremental deep learning},
	elocation-id = {2021.02.26.432552},
	year = {2021},
	doi = {10.1101/2021.02.26.432552},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Deep learning is emerging as a powerful approach for bioimage analysis, but its wider use is limited by the scarcity of annotated data for training. We present ELEPHANT, an interactive platform for cell tracking in 4D that seamlessly integrates annotation, deep learning, and proofreading. ELEPHANT{\textquoteright}s user interface supports cycles of incremental learning starting from sparse annotations, yielding accurate, user-validated cell lineages with a modest investment in time and effort.Competing Interest StatementKS is employed part-time by LPIXEL Inc.},
	URL = {https://www.biorxiv.org/content/early/2021/02/26/2021.02.26.432552},
	eprint = {https://www.biorxiv.org/content/early/2021/02/26/2021.02.26.432552.full.pdf},
	journal = {bioRxiv}
}
```

<div style="text-align: right"><a href="https://www.biorxiv.org/highwire/citation/1813952/bibtext">download as .bib file</a></div>

## License

[BSD-2-Clause](LICENSE)