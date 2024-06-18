#! /usr/bin/env bash

echo "set RUN_ON_FLASK"
export RUN_ON_FLASK=""

echo "#!/usr/bin/env bash\necho RabbitMQ is disabled." > /rabbitmq.sh