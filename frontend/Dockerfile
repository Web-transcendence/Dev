FROM node:lts-alpine3.20

RUN apk add --no-cache openssl

RUN adduser -D -s /bin/sh -u 1001 app
USER app

WORKDIR /app

RUN mkdir secure -p

RUN openssl req -x509 -newkey rsa:4096 -nodes  \
    -out secure/cert.pem \
    -keyout secure/key.pem \
    -days 365 \
    -subj "/C=FR/ST=IDF/L=Lyon/0=42/OU=42/CN=grebrune.42.fr/UID=grebrune"


COPY --chown=app:app *.json .
COPY --chown=app:app src/*.ts src/
COPY --chown=app:app src/serverSide/*.ts src/serverSide/
COPY --chown=app:app src/clientSide/*.ts src/clientSide/
COPY --chown=app:app public/ public/



RUN npm install
RUN npm install @supabase/supabase-js
RUN npm install @fastify/rate-limit


CMD ["npm", "run", "dev"]
