FROM node:lts-alpine3.20

RUN adduser -D -s /bin/sh -u 1001 app
USER app

WORKDIR /app

COPY --chown=app:app *.json .
COPY --chown=app:app src/*.ts src/


RUN npm install

CMD ["npm", "run", "dev"]
