FROM alpine:3.18

RUN apk add --no-cache make g++ boost-dev

RUN adduser -D -s /bin/sh -u 1001 app
USER app

WORKDIR /app

COPY --chown=app:app src/core core

RUN	make -C core/AiServer
RUN	make -C core/Client

# CMD ["tail", "-f", "/dev/null"]

CMD ./core/Client/myFactory "http://127.0.0.1:4443" && ./core/AiServer/myDQN
