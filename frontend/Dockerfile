FROM node:lts-alpine3.20


WORKDIR /app

COPY *.json .
COPY postcss.config.mjs .
COPY tsup.config.ts .
COPY src/*.ts src/
COPY public .


RUN npm install
RUN npx @tailwindcss/cli -i .public/styles.css -o ./public/output.css

CMD ["npm", "run", "dev"]
