.PHONY: help rebuild build launch launch-no-rabbitmq bash bashroot notebook warmup rabbitmq-vhosts test apptainer-build apptainer-init apptainer-launch apptainer-shell apptainer-create-rabbitmq-user apptainer-stop

help:
	@cat Makefile

ELEPHANT_GPU?=all
ELEPHANT_WORKSPACE?=${PWD}/workspace
ELEPHANT_IMAGE_NAME?=elephant-server:0.6.0-dev
ELEPHANT_NVIDIA_GID?=$$(ls -n /dev/nvidia0 2>/dev/null | awk '{print $$4}')
ELEPHANT_DOCKER?=docker
ELEPHANT_RABBITMQ_HOST?=localhost
ELEPHANT_RABBITMQ_NODENAME?=rabbit@$(ELEPHANT_RABBITMQ_HOST)
ELEPHANT_RABBITMQ_NODE_PORT?=5672
ELEPHANT_RABBITMQ_MANAGEMENT_PORT?=15672
ELEPHANT_RABBITMQ_VIRTUAL_HOST?=/
ELEPHANT_RABBITMQ_USER?=user
ELEPHANT_RABBITMQ_PASSWORD?=user
ELEPHANT_RABBITMQ_PID_FILE?=/var/lib/rabbitmq/mnesia/rabbitmq.pid
ELEPHANT_REDIS_PORT?=6379
ELEPHANT_HTTP_PORT?=8080
ELEPHANT_NOTEBOOK_PORT?=8888
ELEPHANT_BATCH_ID?=0

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
	$(ELEPHANT_DOCKER) run -it --rm $(GPU_ARG) --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace \
	-p $(ELEPHANT_HTTP_PORT):$(ELEPHANT_HTTP_PORT) \
	-p $(ELEPHANT_RABBITMQ_NODE_PORT):$(ELEPHANT_RABBITMQ_NODE_PORT) \
	-p $(ELEPHANT_RABBITMQ_MANAGEMENT_PORT):$(ELEPHANT_RABBITMQ_MANAGEMENT_PORT) \
	-e LOCAL_UID=$(shell id -u) \
	-e LOCAL_GID=$(shell id -g) \
	-e NVIDIA_GID=$(ELEPHANT_NVIDIA_GID) \
	-e RABBITMQ_HOST=$(ELEPHANT_RABBITMQ_HOST) \
	-e RABBITMQ_NODENAME=$(ELEPHANT_RABBITMQ_NODENAME) \
	-e RABBITMQ_NODE_PORT=$(ELEPHANT_RABBITMQ_NODE_PORT) \
	-e RABBITMQ_MANAGEMENT_PORT=$(ELEPHANT_RABBITMQ_MANAGEMENT_PORT) \
	-e RABBITMQ_VIRTUAL_HOST=$(ELEPHANT_RABBITMQ_VIRTUAL_HOST) \
	-e RABBITMQ_USER=$(ELEPHANT_RABBITMQ_USER) \
	-e RABBITMQ_PASSWORD=$(ELEPHANT_RABBITMQ_PASSWORD) \
	-e RABBITMQ_PID_FILE=$(ELEPHANT_RABBITMQ_PID_FILE) \
	-e ELEPHANT_REDIS_PORT=$(ELEPHANT_REDIS_PORT) \
	-e ELEPHANT_HTTP_PORT=$(ELEPHANT_HTTP_PORT) \
	-e ELEPHANT_BATCH_ID=$(ELEPHANT_BATCH_ID) \
	$(ELEPHANT_IMAGE_NAME)

launch-no-rabbitmq: warmup
	$(ELEPHANT_DOCKER) run -it --rm $(GPU_ARG) --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace \
	-v ${PWD}/workspace/datasets/Quail_Day1_Ch1/imgs.zarr:/workspace/datasets/Quail_Day1_Ch1/imgs.zarr \
	-v ${PWD}/workspace/datasets/Fig5-6_Flamindo2/imgs.zarr:/workspace/datasets/Fig5-6_Flamindo2/imgs.zarr \
	-v $(PWD)/app/prestart-disable-rabbitmq.sh:/app/prestart.sh \
	--network rabbitmq \
	-p $(ELEPHANT_HTTP_PORT):$(ELEPHANT_HTTP_PORT) \
	-e LOCAL_UID=$(shell id -u) \
	-e LOCAL_GID=$(shell id -g) \
	-e NVIDIA_GID=$(ELEPHANT_NVIDIA_GID) \
	-e RABBITMQ_HOST=rabbitmq \
	-e RABBITMQ_NODENAME=$(ELEPHANT_RABBITMQ_NODENAME) \
	-e RABBITMQ_NODE_PORT=$(ELEPHANT_RABBITMQ_NODE_PORT) \
	-e RABBITMQ_MANAGEMENT_PORT=$(ELEPHANT_RABBITMQ_MANAGEMENT_PORT) \
	-e RABBITMQ_VIRTUAL_HOST=$(ELEPHANT_RABBITMQ_VIRTUAL_HOST) \
	-e RABBITMQ_USER=$(ELEPHANT_RABBITMQ_USER) \
	-e RABBITMQ_PASSWORD=$(ELEPHANT_RABBITMQ_PASSWORD) \
	-e RABBITMQ_PID_FILE=$(ELEPHANT_RABBITMQ_PID_FILE) \
	-e ELEPHANT_REDIS_PORT=$(ELEPHANT_REDIS_PORT) \
	-e ELEPHANT_HTTP_PORT=$(ELEPHANT_HTTP_PORT) \
	-e ELEPHANT_BATCH_ID=$(ELEPHANT_BATCH_ID) \
	$(ELEPHANT_IMAGE_NAME)

bash: warmup
	$(ELEPHANT_DOCKER) run -it --rm $(GPU_ARG) --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace \
	-v /lustre1/users/sugawara/bigdataserver/00_Kakshine:/00_Kakshine \
	-v ${PWD}/workspace/datasets/Fig5-6_Flamindo2/imgs.zarr:/workspace/datasets/Fig5-6_Flamindo2/imgs.zarr \
	-e LOCAL_UID=$(shell id -u) -e LOCAL_GID=$(shell id -g) -e AS_LOCAL_USER=1 -e NVIDIA_GID=$(ELEPHANT_NVIDIA_GID) \
	--network rabbitmq \
	$(ELEPHANT_IMAGE_NAME) /bin/bash

bashroot: warmup
	$(ELEPHANT_DOCKER) run -it --rm $(GPU_ARG) --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace \
	$(ELEPHANT_IMAGE_NAME) /bin/bash

notebook: warmup
	$(ELEPHANT_DOCKER) run -it --rm $(GPU_ARG) --shm-size=8g -v $(ELEPHANT_WORKSPACE):/workspace \
	-e LOCAL_UID=$(shell id -u) -e LOCAL_GID=$(shell id -g) -e AS_LOCAL_USER=1 -e NVIDIA_GID=$(ELEPHANT_NVIDIA_GID) \
	--network host -p $(ELEPHANT_NOTEBOOK_PORT):$(ELEPHANT_NOTEBOOK_PORT) $(ELEPHANT_IMAGE_NAME) jupyter notebook --no-browser --port=$(ELEPHANT_NOTEBOOK_PORT) --notebook-dir=/workspace

rabbitmq-vhosts:
	$(ELEPHANT_DOCKER) run -it --rm \
	--name rabbitmq \
	--network rabbitmq \
	-p $(ELEPHANT_RABBITMQ_NODE_PORT):$(ELEPHANT_RABBITMQ_NODE_PORT) \
	-p $(ELEPHANT_RABBITMQ_MANAGEMENT_PORT):$(ELEPHANT_RABBITMQ_MANAGEMENT_PORT) \
	-e RABBITMQ_NODENAME=$(ELEPHANT_RABBITMQ_NODENAME) \
	-e RABBITMQ_NODE_PORT=$(ELEPHANT_RABBITMQ_NODE_PORT) \
	-e RABBITMQ_MANAGEMENT_PORT=$(ELEPHANT_RABBITMQ_MANAGEMENT_PORT) \
	-e RABBITMQ_VIRTUAL_HOST=$(ELEPHANT_RABBITMQ_VIRTUAL_HOST) \
	-e RABBITMQ_USER=$(ELEPHANT_RABBITMQ_USER) \
	-e RABBITMQ_PASSWORD=$(ELEPHANT_RABBITMQ_PASSWORD) \
	-e RABBITMQ_PID_FILE=$(ELEPHANT_RABBITMQ_PID_FILE) \
	$(ELEPHANT_IMAGE_NAME) /rabbitmq-vhosts.sh

test:
	$(ELEPHANT_DOCKER) build -t $(ELEPHANT_IMAGE_NAME)-test -f Dockerfile-test . && $(ELEPHANT_DOCKER) image prune -f 
	$(ELEPHANT_DOCKER) run -it --rm $(ELEPHANT_IMAGE_NAME)-test

apptainer-build:
	apptainer build --fakeroot elephant.sif elephant.def

apptainer-init:
	apptainer run --fakeroot --bind $(HOME):/root elephant.sif

apptainer-launch:
	apptainer instance start --nv \
	--bind \
	$(HOME),\
	$(HOME)/.elephant_binds/var/lib:/var/lib,\
	$(HOME)/.elephant_binds/var/log:/var/log,\
	$(HOME)/.elephant_binds/var/run:/var/run,\
	$(HOME)/.elephant_binds/etc/nginx:/etc/nginx,\
	$(ELEPHANT_WORKSPACE):/workspace \
	elephant.sif elephant$(ELEPHANT_BATCH_ID)
	if [ $(ELEPHANT_GPU) = all ]; then \
		apptainer exec \
		--env RABBITMQ_HOST=$(ELEPHANT_RABBITMQ_HOST) \
		--env RABBITMQ_NODENAME=$(ELEPHANT_RABBITMQ_NODENAME) \
		--env RABBITMQ_NODE_PORT=$(ELEPHANT_RABBITMQ_NODE_PORT) \
		--env RABBITMQ_MANAGEMENT_PORT=$(ELEPHANT_RABBITMQ_MANAGEMENT_PORT) \
		--env RABBITMQ_VIRTUAL_HOST=$(ELEPHANT_RABBITMQ_VIRTUAL_HOST) \
		--env RABBITMQ_USER=$(ELEPHANT_RABBITMQ_USER) \
		--env RABBITMQ_PASSWORD=$(ELEPHANT_RABBITMQ_PASSWORD) \
		--env RABBITMQ_PID_FILE=$(ELEPHANT_RABBITMQ_PID_FILE) \
		--env ELEPHANT_REDIS_PORT=$(ELEPHANT_REDIS_PORT) \
		--env ELEPHANT_HTTP_PORT=$(ELEPHANT_HTTP_PORT) \
		--env ELEPHANT_BATCH_ID=$(ELEPHANT_BATCH_ID) \
		instance://elephant$(ELEPHANT_BATCH_ID) /start.sh; \
	else \
		apptainer exec \
		--env CUDA_VISIBLE_DEVICES=$(ELEPHANT_GPU) \
		--env RABBITMQ_HOST=$(ELEPHANT_RABBITMQ_HOST) \
		--env RABBITMQ_NODENAME=$(ELEPHANT_RABBITMQ_NODENAME) \
		--env RABBITMQ_NODE_PORT=$(ELEPHANT_RABBITMQ_NODE_PORT) \
		--env RABBITMQ_MANAGEMENT_PORT=$(ELEPHANT_RABBITMQ_MANAGEMENT_PORT) \
		--env RABBITMQ_VIRTUAL_HOST=$(ELEPHANT_RABBITMQ_VIRTUAL_HOST) \
		--env RABBITMQ_USER=$(ELEPHANT_RABBITMQ_USER) \
		--env RABBITMQ_PASSWORD=$(ELEPHANT_RABBITMQ_PASSWORD) \
		--env RABBITMQ_PID_FILE=$(ELEPHANT_RABBITMQ_PID_FILE) \
		--env ELEPHANT_REDIS_PORT=$(ELEPHANT_REDIS_PORT) \
		--env ELEPHANT_HTTP_PORT=$(ELEPHANT_HTTP_PORT) \
		--env ELEPHANT_BATCH_ID=$(ELEPHANT_BATCH_ID) \
		instance://elephant$(ELEPHANT_BATCH_ID) /start.sh; \
	fi

apptainer-shell:
	if [ $(ELEPHANT_GPU) = all ]; then \
		apptainer shell --fakeroot --nv \
		--bind $(HOME) \
		--bind $(HOME)/.elephant_binds/var/lib:/var/lib \
		--bind $(HOME)/.elephant_binds/var/log:/var/log \
		--bind $(HOME)/.elephant_binds/var/run:/var/run \
		--bind $(HOME)/.elephant_binds/etc/nginx:/etc/nginx \
		elephant.sif; \
	else \
		apptainer shell --fakeroot --nv \
		--env CUDA_VISIBLE_DEVICES=$(ELEPHANT_GPU) \
		--bind $(HOME) \
		--bind $(HOME)/.elephant_binds/var/lib:/var/lib \
		--bind $(HOME)/.elephant_binds/var/log:/var/log \
		--bind $(HOME)/.elephant_binds/var/run:/var/run \
		--bind $(HOME)/.elephant_binds/etc/nginx:/etc/nginx \
		elephant.sif; \
	fi

apptainer-stop:
	apptainer instance stop elephant$(ELEPHANT_BATCH_ID)
