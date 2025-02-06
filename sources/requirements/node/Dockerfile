FROM node:23-alpine3.20 AS builder-back

# Copy sources
WORKDIR /app
COPY ./back .

# Install deps to build sources
RUN npm ci

# Build project with TS
RUN npm run build:back

FROM node:23-alpine3.20 AS builder-front

# Copy sources
WORKDIR /app
COPY ./front .

# Install deps to build sources
RUN npm ci

# Build project with TS
RUN npm run build:front

FROM node:23-alpine3.20 AS runner

# Copy backend
COPY --from=builder-back /app/dist /app
# Copy frontend
COPY --from=builder-front /app/dist /app/front
# Copy package infos
COPY ./back/package.json /app/
COPY ./back/package-lock.json /app/
# Copy html views
COPY ./back/src/views /app/views

# Set the working directory inside the container
WORKDIR /app
ENV TRANS_FRONT_PATH="./front"
ENV TRANS_VIEWS_PATH="./views"

# Install dependencies
RUN npm ci --omit dev

# Expose the Fastify server port
EXPOSE 8081

# Start the Fastify server
CMD ["node", "index.js"]
