#! /usr/bin/env bash
set -x
mkdir -p /etc/ssl/mycerts
chmod 755 /etc/ssl/mycerts
cd /tmp
git clone https://github.com/rabbitmq/tls-gen tls-gen
cd tls-gen/basic
make CN=localhost
cp -a /tmp/tls-gen/basic/result/*.pem /etc/ssl/mycerts/
chown -R $UWSGI_UID:$UWSGI_GID /etc/ssl/mycerts
chmod 644 /etc/ssl/mycerts/*
rm -rf /tmp/tls-gen