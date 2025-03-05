FROM node:lts-alpine3.20

WORKDIR /app

COPY *.json .
COPY src/*.ts src/



RUN npm install

CMD ["npm", "run", "dev"]
