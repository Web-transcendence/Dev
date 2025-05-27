#!/bin/sh
# Fix permissions only if needed
chmod g+w /app/data 2>/dev/null || true
exec "$@"