FROM node:23-alpine3.20 AS builder-back

# Copy sources
WORKDIR /app
COPY . .

# Install deps to build sources
RUN ls && npm ci

# Build project with TS
RUN npm run build

FROM node:23-alpine3.20 AS runner

# Copy package.json and package-lock.json first (for efficient caching)
COPY --from=builder-back /app/dist /app
COPY package.json /app/
COPY package-lock.json /app/

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
RUN npm ci --omit dev

# Build Tailwind CSS
#RUN npx tailwindcss -i ./src/styles.css -o ./dist/output.css

# Expose the Fastify server port
EXPOSE 8081

# Start the Fastify server
CMD ["node", "index.js"]
