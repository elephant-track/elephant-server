.PHONY: help rebuild build launch bash bashroot warmup test singularity-build singularity-launch singularity-stop

help:
	@cat Makefile

ELEPHANT_GPU?=all
ELEPHANT_WORKSPACE?=${PWD}/workspace
ELEPHANT_IMAGE_NAME?=elephant-server:0.4.0
ELEPHANT_NVIDIA_GID?=$$(ls -n /dev/nvidia0 2>/dev/null | awk '{print $$4}')
ELEPHANT_DOCKER?=docker

rebuild:
	@IMAGEID=$$($(ELEPHANT_DOCKER) images -q $(ELEPHANT_IMAGE_NAME)); \
	$(ELEPHANT_DOCKER) build --no-cache -t $(ELEPHANT_IMAGE_NAME) .; \
	if [ -n "$$IMAGEID" ]; then \
		$(ELEPHANT_DOCKER) rmi $$IMAGEID; \
	fi

build:
	$(ELEPHANT_DOCKER) build -t $(ELEPHANT_IMAGE_NAME) . && $(ELEPHANT_DOCKER) image prune -f

stop:
	@CONTAINERID=$$($(ELEPHANT_DOCKER) ps -aq --filter ancestor=$(ELEPHANT_IMAGE_NAME)); \
	if [ -n "$$CONTAINERID" ]; then \
		$(ELEPHANT_DOCKER) stop $$CONTAINERID; \
	fi

warmup:
	$(ELEPHANT_DOCKER) run -it --rm --gpus all $(ELEPHANT_IMAGE_NAME) echo "warming up..."

launch: warmup
	$(ELEPHANT_DOCKER) run -it --rm --gpus '"device=$(ELEPHANT_GPU)"' --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace -p 8080:80 -p 5672:5672 \
	-e LOCAL_UID=$(shell id -u) -e LOCAL_GID=$(shell id -g) -e NVIDIA_GID=$(ELEPHANT_NVIDIA_GID) \
	$(ELEPHANT_IMAGE_NAME)

bash: warmup
	$(ELEPHANT_DOCKER) run -it --rm --gpus '"device=$(ELEPHANT_GPU)"' --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace \
	-e LOCAL_UID=$(shell id -u) -e LOCAL_GID=$(shell id -g) -e AS_LOCAL_USER=1 -e NVIDIA_GID=$(ELEPHANT_NVIDIA_GID) \
	$(ELEPHANT_IMAGE_NAME) /bin/bash

bashroot:
	$(ELEPHANT_DOCKER) run -it --rm --gpus '"device=$(ELEPHANT_GPU)"' --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace \
	$(ELEPHANT_IMAGE_NAME) /bin/bash

test:
	$(ELEPHANT_DOCKER) build -t $(ELEPHANT_IMAGE_NAME)-test -f Dockerfile-test . && $(ELEPHANT_DOCKER) image prune -f 
	$(ELEPHANT_DOCKER) run -it --rm $(ELEPHANT_IMAGE_NAME)-test

singularity-build:
	singularity build --fakeroot elephant.sif elephant.def
	singularity run --fakeroot elephant.sif

singularity-launch:
	singularity instance start --nv --bind $(HOME)/.elephant_binds/var/lib:/var/lib,$(HOME)/.elephant_binds/var/log:/var/log,$(HOME)/.elephant_binds/var/run:/var/run,$(ELEPHANT_WORKSPACE):/workspace elephant.sif elephant
	SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(ELEPHANT_GPU) singularity exec instance://elephant /start.sh

singularity-stop:
	singularity instance stop elephant