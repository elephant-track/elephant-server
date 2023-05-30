.PHONY: help rebuild build launch bash bashroot notebook warmup test apptainer-build apptainer-launch apptainer-shell apptainer-stop

help:
	@cat Makefile

ELEPHANT_GPU?=all
ELEPHANT_WORKSPACE?=${PWD}/workspace
ELEPHANT_IMAGE_NAME?=elephant-server:0.5.0
ELEPHANT_NVIDIA_GID?=$$(ls -n /dev/nvidia0 2>/dev/null | awk '{print $$4}')
ELEPHANT_DOCKER?=docker
ELEPHANT_REDIS_PORT?=6379

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
	$(eval GPU_ARG:=$(shell \
	if [ -n "$(ELEPHANT_NVIDIA_GID)" ] && [ -n "$(ELEPHANT_GPU)" ]; then \
		VAR=$$(echo --gpus '"device=$(ELEPHANT_GPU)"'); \
	fi;\
	echo $$VAR))
	@if [ -n "$(GPU_ARG)" ]; then \
		$(ELEPHANT_DOCKER) run -it --rm $(GPU_ARG) $(ELEPHANT_IMAGE_NAME) echo "warming up GPU..."; \
	else \
		echo "CPU mode..."; \
	fi

launch: warmup
	$(ELEPHANT_DOCKER) run -it --rm $(GPU_ARG) --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace -p 8080:80 -p 5672:5672 \
	-e LOCAL_UID=$(shell id -u) -e LOCAL_GID=$(shell id -g) -e NVIDIA_GID=$(ELEPHANT_NVIDIA_GID) -e ELEPHANT_REDIS_PORT=$(ELEPHANT_REDIS_PORT)\
	$(ELEPHANT_IMAGE_NAME)

bash: warmup
	$(ELEPHANT_DOCKER) run -it --rm $(GPU_ARG) --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace \
	-e LOCAL_UID=$(shell id -u) -e LOCAL_GID=$(shell id -g) -e AS_LOCAL_USER=1 -e NVIDIA_GID=$(ELEPHANT_NVIDIA_GID) \
	$(ELEPHANT_IMAGE_NAME) /bin/bash

bashroot: warmup
	$(ELEPHANT_DOCKER) run -it --rm $(GPU_ARG) --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace \
	$(ELEPHANT_IMAGE_NAME) /bin/bash

notebook: warmup
	$(ELEPHANT_DOCKER) run -it --rm $(GPU_ARG) --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace \
	-e LOCAL_UID=$(shell id -u) -e LOCAL_GID=$(shell id -g) -e AS_LOCAL_USER=1 -e NVIDIA_GID=$(ELEPHANT_NVIDIA_GID) \
	--network host -p 8888:8888 $(ELEPHANT_IMAGE_NAME) jupyter notebook --no-browser --notebook-dir=/workspace

test:
	$(ELEPHANT_DOCKER) build -t $(ELEPHANT_IMAGE_NAME)-test -f Dockerfile-test . && $(ELEPHANT_DOCKER) image prune -f 
	$(ELEPHANT_DOCKER) run -it --rm $(ELEPHANT_IMAGE_NAME)-test

apptainer-build:
	apptainer build --fakeroot elephant.sif elephant.def
	apptainer run --fakeroot --bind $(HOME):/root elephant.sif

apptainer-launch:
	apptainer instance start --nv --bind $(HOME),$(HOME)/.elephant_binds/var/lib:/var/lib,$(HOME)/.elephant_binds/var/log:/var/log,$(HOME)/.elephant_binds/var/run:/var/run,$(HOME)/.elephant_binds/etc/nginx:/etc/nginx,$(HOME)/.elephant_binds/etc/rabbitmq:/etc/rabbitmq,$(ELEPHANT_WORKSPACE):/workspace elephant.sif elephant
	if [ $(ELEPHANT_GPU) = all ]; then \
		apptainer exec --env ELEPHANT_REDIS_PORT=$(ELEPHANT_REDIS_PORT) instance://elephant /start.sh; \
	else \
		apptainer exec --env CUDA_VISIBLE_DEVICES=$(ELEPHANT_GPU),ELEPHANT_REDIS_PORT=$(ELEPHANT_REDIS_PORT) instance://elephant /start.sh; \
	fi 

apptainer-shell:
	apptainer shell --env ELEPHANT_REDIS_PORT=$(ELEPHANT_REDIS_PORT) --bind $(HOME),$(HOME)/.elephant_binds/var/lib:/var/lib,$(HOME)/.elephant_binds/var/log:/var/log,$(HOME)/.elephant_binds/var/run:/var/run,$(HOME)/.elephant_binds/etc/nginx:/etc/nginx,$(HOME)/.elephant_binds/etc/rabbitmq:/etc/rabbitmq elephant.sif

apptainer-stop:
	apptainer instance stop elephant
