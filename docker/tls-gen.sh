#! /usr/bin/env bash
set -x
mkdir -p /etc/ssl/mycerts
cd /tmp
git clone https://github.com/rabbitmq/tls-gen tls-gen
cd tls-gen/basic
make CN=$RABBITMQ_USER
cp -a /tmp/tls-gen/basic/result/*.pem /etc/ssl/mycerts/
chown -R $UWSGI_UID:$UWSGI_GID /etc/ssl/mycerts
rm -rf /tmp/tls-gen