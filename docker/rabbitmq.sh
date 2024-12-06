#!/bin/sh
# https://stackoverflow.com/questions/30747469/how-to-add-initial-users-when-starting-a-rabbitmq-docker-container

# Create Rabbitmq user
( rabbitmqctl wait --timeout 60 $RABBITMQ_PID_FILE ; \
rabbitmqctl add_user $RABBITMQ_USER $RABBITMQ_PASSWORD 2>/dev/null ; \
rabbitmqctl set_user_tags $RABBITMQ_USER administrator ; \
rabbitmqctl set_permissions -p / $RABBITMQ_USER  ".*" ".*" ".*" ; \
echo "*** User '$RABBITMQ_USER' with password '$RABBITMQ_PASSWORD' completed. ***" ; \
echo "*** RabbitMQ server is working at $RABBITMQ_NODENAME:$RABBITMQ_NODE_PORT ***"; \
echo "*** Log in the WebUI at port $RABBITMQ_MANAGEMENT_PORT (example: http://localhost:$RABBITMQ_MANAGEMENT_PORT) ***") &

# enable rabbitmq_management
rabbitmq-plugins enable rabbitmq_management

echo "management.tcp.port = $RABBITMQ_MANAGEMENT_PORT" > /etc/rabbitmq/rabbitmq.conf
if [ "$RABBITMQ_USE_SSL" = "true" ]; then
  echo "*** RabbitMQ use SSL: $RABBITMQ_USE_SSL. ***"
  echo "listeners.ssl.default = $RABBITMQ_NODE_PORT" >> /etc/rabbitmq/rabbitmq.conf
  echo "ssl_options.cacertfile = /etc/ssl/mycerts/ca_certificate.pem" >> /etc/rabbitmq/rabbitmq.conf
  echo "ssl_options.certfile = /etc/ssl/mycerts/server_localhost_certificate.pem" >> /etc/rabbitmq/rabbitmq.conf
  echo "ssl_options.keyfile = /etc/ssl/mycerts/server_localhost_key.pem" >> /etc/rabbitmq/rabbitmq.conf
  echo "ssl_options.verify = $RABBITMQ_SSL_VERIFY" >> /etc/rabbitmq/rabbitmq.conf
  echo "ssl_options.fail_if_no_peer_cert = $RABBITMQ_SSL_FAIL_IF_NO_PEER_CERT" >> /etc/rabbitmq/rabbitmq.conf
fi

# $@ is used to pass arguments to the rabbitmq-server command.
# For example if you use it like this: docker run -d rabbitmq arg1 arg2,
# it will be as you run in the container rabbitmq-server arg1 arg2 
rabbitmq-server $@