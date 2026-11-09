#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
exec /opt/instance-tools/bin/caddy run --config /opt/dewatermark/config/console.Caddyfile --adapter caddyfile
