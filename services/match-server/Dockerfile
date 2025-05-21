FROM node:lts-alpine3.20
RUN apk add curl

RUN apk add --no-cache openssl

RUN adduser -D -s /bin/sh -u 1001 app
USER app

WORKDIR /app

RUN mkdir secure -p

RUN openssl req -x509 -newkey rsa:4096 -nodes  \
    -out secure/cert.pem \
    -keyout secure/key.pem \
    -days 365 \
    -subj "/C=FR/ST=IDF/L=Lyon/0=42/OU=42/CN=beroy.42.fr/UID=beroy"


COPY --chown=app:app *.json .
COPY --chown=app:app src/*.ts src/

EXPOSE 4443 

RUN npm install

CMD ["npm", "run", "dev"]
