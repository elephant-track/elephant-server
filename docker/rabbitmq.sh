#!/bin/sh
# https://stackoverflow.com/questions/30747469/how-to-add-initial-users-when-starting-a-rabbitmq-docker-container

# Create Rabbitmq user
( rabbitmqctl wait --timeout 60 $RABBITMQ_PID_FILE ; \
rabbitmqctl add_user $RABBITMQ_USER $RABBITMQ_PASSWORD 2>/dev/null ; \
rabbitmqctl set_user_tags $RABBITMQ_USER administrator ; \
rabbitmqctl set_permissions -p / $RABBITMQ_USER  ".*" ".*" ".*" ; \
echo "*** User '$RABBITMQ_USER' with password '$RABBITMQ_PASSWORD' completed. ***" ; \
echo "*** RabbitMQ server is working at $RABBITMQ_NODENAME:$RABBITMQ_NODE_PORT$RABBITMQ_VIRTUAL_HOST ***"; \
echo "*** Log in the WebUI at port $RABBITMQ_MANAGEMENT_PORT (example: http://localhost:$RABBITMQ_MANAGEMENT_PORT) ***") &

# enable rabbitmq_management
rabbitmq-plugins enable rabbitmq_management

echo "management.tcp.port = $RABBITMQ_MANAGEMENT_PORT" > /etc/rabbitmq/rabbitmq.conf

# $@ is used to pass arguments to the rabbitmq-server command.
# For example if you use it like this: docker run -d rabbitmq arg1 arg2,
# it will be as you run in the container rabbitmq-server arg1 arg2 
rabbitmq-server $@