FROM node:lts-alpine3.20

RUN apk add --no-cache openssl

WORKDIR /app

RUN mkdir secure -p

RUN openssl req -x509 -newkey rsa:4096 -nodes  \
    -out secure/cert.pem \
    -keyout secure/key.pem \
    -days 365 \
    -subj "/C=FR/ST=IDF/L=Lyon/0=42/OU=42/CN=grebrune.42.fr/UID=grebrune"

COPY *.json .
COPY src/*.ts src/
COPY src/serverSide/*.ts src/serverSide/
COPY src/clientSide/*.ts src/clientSide/
COPY public/ public/

RUN npm install

CMD ["npm", "run", "prod"]
