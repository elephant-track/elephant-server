FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
# Modified from https://github.com/tiangolo/uwsgi-nginx-flask-docker (Apache license)

LABEL maintainer="Ko Sugawara <ko.sugawara@ens-lyon.fr>"

# Install requirements
RUN set -x \
    && apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y \
    nginx \
    redis-server \
    supervisor \
    ca-certificates \
    curl \
    gnupg \
    gosu \
    gettext-base && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Install and set up RabbbitMQ
COPY docker/install-rabbitmq.sh /tmp/install-rabbitmq.sh
RUN chmod +x /tmp/install-rabbitmq.sh && /tmp/install-rabbitmq.sh && rm /tmp/install-rabbitmq.sh
EXPOSE 5672
ENV RABBITMQ_USER user
ENV RABBITMQ_PASSWORD user
ENV RABBITMQ_PID_FILE /var/lib/rabbitmq/mnesia/rabbitmq.pid
COPY docker/rabbitmq.sh /rabbitmq.sh
RUN chmod +x /rabbitmq.sh

# Set up nginx
COPY docker/nginx.conf.template /etc/nginx/nginx.conf.template
EXPOSE 80 443
RUN groupadd -r nginx && useradd -r -g nginx nginx
# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

# install cuda-related modules
RUN conda install -y -c nvidia cuda-nvcc=12.3
RUN python -m pip install --no-deps nvidia-cudnn-cu12==8.9.2.26

# install pypi packages
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt
# RUN pip install memory_profiler line_profiler
# RUN pip install --no-deps stardist==0.8.3 csbdeep==0.7.2 numba==0.56.0 llvmlite==0.39.0 natsort==8.1.0

# Copy the base uWSGI ini file to enable default dynamic uwsgi process number
COPY docker/uwsgi.ini /etc/uwsgi/uwsgi.ini

# Custom Supervisord config
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy start.sh script that will check for a /app/prestart.sh script and run it before starting the app
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

# Make /app/* available to be imported by Python globally to better support several use cases like Alembic migrations.
ENV PYTHONPATH=/app

# Add scripts
COPY ./script /opt/elephant/script

# Add Flask app
COPY ./app /app
WORKDIR /app

# Copy the entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

COPY ./elephant-core /tmp/elephant-core
RUN pip install -U /tmp/elephant-core && rm -rf /tmp/elephant-core

# Run the start script provided by the parent image tiangolo/uwsgi-nginx.
# It will check for an /app/prestart.sh script (e.g. for migrations)
# And then will start Supervisor, which in turn will start Nginx and uWSGI
CMD ["/start.sh"]
