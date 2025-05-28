FROM node:lts-alpine3.20

WORKDIR /app

RUN mkdir db

COPY *.json .
COPY src/*.ts src/
COPY resources/*.json resources/

RUN npm install

CMD ["npm", "run", "prod"]
