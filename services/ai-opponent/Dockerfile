FROM alpine:3.18

RUN adduser -D -s /bin/sh -u 1001 app
USER app

COPY src/core/AiServer/myDQN /app/myDQN
COPY weights.json /app/weights.json
COPY src/core/Client/myFactory /app/myFactory
COPY supervisord.conf /etc/supervisord.conf
COPY entrypoint.sh /app/entrypoint.sh



WORKDIR /app

RUN chmod +x /app/entrypoint.sh

CMD ["./entrypoint.sh"]
