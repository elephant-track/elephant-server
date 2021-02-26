FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
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
    gosu && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install conda modules
RUN conda install --override-channels -c main -c conda-forge \
    h5py=2.10.0 \
    flask=1.1.2 \
    flask-redis=0.4.0 \
    libiconv=1.15 \
    pika=1.1.0 \
    scikit-learn=0.23.1 \
    scikit-image=0.17.2 \
    scipy=1.4.1 \
    tensorboardX=2.1 \
    tqdm=4.48.2 \
    uwsgi=2.0.18 \
    zarr=2.4.0

# Install and set up RabbbitMQ
RUN curl -fsSL https://github.com/rabbitmq/signing-keys/releases/download/2.0/rabbitmq-release-signing-key.asc | apt-key add - && \
    apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y \
    apt-transport-https && \
    echo 'deb http://dl.bintray.com/rabbitmq-erlang/debian bionic erlang-22.x\ndeb https://dl.bintray.com/rabbitmq/debian bionic main' >> /etc/apt/sources.list.d/bintray.rabbitmq.list && \
    echo 'Package: erlang*\nPin: release o=Bintray\nPin-Priority: 1000' >> /etc/apt/preferences.d/erlang && \
    apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y \
    rabbitmq-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
EXPOSE 5672
ENV RABBITMQ_USER user
ENV RABBITMQ_PASSWORD user
ENV RABBITMQ_PID_FILE /var/lib/rabbitmq/mnesia/rabbitmq.pid
COPY docker/rabbitmq.sh /rabbitmq.sh
RUN chmod +x /rabbitmq.sh

# Set up nginx
COPY docker/nginx.conf /etc/nginx/nginx.conf
EXPOSE 80 443
RUN groupadd nginx && useradd -g nginx nginx
# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

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

# Install elephant core
COPY ./elephant-core /tmp/elephant-core
RUN pip install -U /tmp/elephant-core && rm -rf /tmp/elephant-core

# Copy the entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Run the start script provided by the parent image tiangolo/uwsgi-nginx.
# It will check for an /app/prestart.sh script (e.g. for migrations)
# And then will start Supervisor, which in turn will start Nginx and uWSGI
CMD ["/start.sh"]
