FROM node:lts-alpine3.20

RUN adduser -D -s /bin/sh -u 1001 app
USER app

WORKDIR /app

COPY --chown=app:app *.json .
COPY --chown=app:app src/*.ts src/
COPY --chown=app:app public/ public/


RUN npm install
RUN npx @tailwindcss/cli -i ./public/styles.css -o ./public/output.css

CMD ["npm", "run", "dev"]
