#!/usr/bin/env bash
/etc/init.d/rabbitmq-server start
rabbitmqctl add_user user user 2>/dev/null
rabbitmqctl set_user_tags user administrator
rabbitmqctl set_permissions -p / user  ".*" ".*" ".*"
/etc/init.d/rabbitmq-server stop