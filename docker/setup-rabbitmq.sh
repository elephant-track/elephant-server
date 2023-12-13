#!/usr/bin/env bash
/etc/init.d/rabbitmq-server start
rabbitmqctl add_user $RABBITMQ_USER $RABBITMQ_PASSWORD 2>/dev/null
rabbitmqctl set_user_tags $RABBITMQ_USER administrator
rabbitmqctl set_permissions -p / $RABBITMQ_USER  ".*" ".*" ".*"
/etc/init.d/rabbitmq-server stop
echo "*** User '$RABBITMQ_USER' with password '$RABBITMQ_PASSWORD' completed. ***"
echo "*** RabbitMQ server is working at $RABBITMQ_NODENAME:$RABBITMQ_NODE_PORT ***"
echo "*** Log in the WebUI at port $RABBITMQ_MANAGEMENT_PORT (example: http://localhost:$RABBITMQ_MANAGEMENT_PORT) ***"

# enable rabbitmq_management
rabbitmq-plugins enable rabbitmq_management

echo "management.tcp.port = $RABBITMQ_MANAGEMENT_PORT" > /etc/rabbitmq/rabbitmq.conf