.PHONY: help rebuild build launch bash bashroot

help:
	@cat Makefile

ELEPHANT_GPU?=0
ELEPHANT_WORKSPACE?=${PWD}/workspace
ELEPHANT_IMAGE_NAME?=elephant-server:0.1.0
ELEPHANT_NVIDIA_GID?=$$(ls -n /dev/nvidia$(ELEPHANT_GPU) | awk '{print $$4}')
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

launch:
	$(ELEPHANT_DOCKER) run -it --rm --gpus device=$(ELEPHANT_GPU) -v $(ELEPHANT_WORKSPACE):/workspace -p 8080:80 -p 5672:5672 \
	-e LOCAL_UID=$(shell id -u) -e LOCAL_GID=$(shell id -g) -e NVIDIA_GID=$(ELEPHANT_NVIDIA_GID) \
	$(ELEPHANT_IMAGE_NAME)

bash:
	$(ELEPHANT_DOCKER) run -it --rm --gpus device=$(ELEPHANT_GPU) -v $(ELEPHANT_WORKSPACE):/workspace \
	-e LOCAL_UID=$(shell id -u) -e LOCAL_GID=$(shell id -g) -e AS_LOCAL_USER=1 -e NVIDIA_GID=$(ELEPHANT_NVIDIA_GID) \
	$(ELEPHANT_IMAGE_NAME) /bin/bash

bashroot:
	$(ELEPHANT_DOCKER) run -it --rm --gpus device=$(ELEPHANT_GPU) -v $(ELEPHANT_WORKSPACE):/workspace \
	$(ELEPHANT_IMAGE_NAME) /bin/bash
