FROM node:23-alpine3.20 AS builder-back

# Copy sources
WORKDIR /app
COPY . .

# Install deps to build sources
RUN npm ci

FROM node:23-alpine3.20 AS runner

# Copy package.json and package-lock.json first (for efficient caching)
COPY --from=builder-back /app /app

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
RUN npm ci --omit dev

# Copy the rest of the application files
COPY . .

# Expose the Fastify server port
EXPOSE 8081

# Start the Fastify server
CMD ["node", "server.js"]
