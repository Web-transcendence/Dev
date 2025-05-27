FROM alpine:3.18

RUN apk add --no-cache make g++ boost-dev

RUN adduser -D -s /bin/sh -u 1001 app
USER app

WORKDIR /app

COPY --chown=app:app src/core core
COPY --chown=app:app tools/binariesWrapper.sh binariesWrapper.sh

RUN	chmod +x binariesWrapper.sh

# RUN	make -C core/AiServer
RUN	make -C core/Client

ENTRYPOINT ["./binariesWrapper.sh"]
