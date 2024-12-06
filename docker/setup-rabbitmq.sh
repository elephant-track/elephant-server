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
if [ "$RABBITMQ_USE_SSL" = "true" ]; then
  echo "*** RabbitMQ use SSL: $RABBITMQ_USE_SSL. ***"
  bash /docker/tls-gen.sh
  echo "listeners.ssl.default = $RABBITMQ_NODE_PORT" > /etc/rabbitmq/rabbitmq.conf  # TLS用のポートを指定
  echo "ssl_options.cacertfile = /etc/ssl/mycerts/ca_certificate.pem" > /etc/rabbitmq/rabbitmq.conf
  echo "ssl_options.certfile = /etc/ssl/mycerts/server_${RABBITMQ_USER}_certificate.pem" > /etc/rabbitmq/rabbitmq.conf
  echo "ssl_options.keyfile = /etc/ssl/mycerts/server_${RABBITMQ_USER}_key.pem" > /etc/rabbitmq/rabbitmq.conf
  echo "ssl_options.verify = $RABBITMQ_SSL_VERIFY" > /etc/rabbitmq/rabbitmq.conf  # クライアント認証を有効にする場合
  echo "ssl_options.fail_if_no_peer_cert = $RABBITMQ_SSL_FAIL_IF_NO_PEER_CERT" > /etc/rabbitmq/rabbitmq.conf  # 必須クライアント証明書
fi