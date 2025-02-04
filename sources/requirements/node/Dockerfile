#FROM node:23-alpine3.20 AS builder-back
#
## Copy sources
#COPY . /app
#WORKDIR /app
## Install deps to build sources
## Build sources
##RUN npm run build
#
#FROM node:23-alpine3.20 AS runner
#
#
##RUN npm install fastify
#COPY package*.json ./
## Copy result of build (small app)
#COPY --from=builder-back /app /app
#COPY . .
## Metadata
#EXPOSE 8081
#WORKDIR /app
#ENTRYPOINT ["node", "/app/server.js"]

# Use an official Node.js runtime as the base image
FROM node:18-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json first (for efficient caching)
COPY package*.json .

# Ensure "type": "module" is present in package.json
RUN echo '{ "type": "module" }' > package.json

# Install dependencies
RUN npm install
RUN npm install fastify

# Copy the rest of the application files
COPY . .

# Expose the Fastify server port
EXPOSE 8081

# Start the Fastify server
CMD ["node", "server.js"]
