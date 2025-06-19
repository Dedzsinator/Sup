# Multi-stage Dockerfile for React Native Frontend
FROM node:18-alpine AS deps

# Set working directory
WORKDIR /app

# Install global dependencies
RUN npm install -g @expo/cli

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && \
    npm cache clean --force

# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Install global dependencies
RUN npm install -g @expo/cli

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev)
RUN npm ci

# Copy source code
COPY . .

# Build for web
RUN npm run build:web || npx expo export:web

# Production stage
FROM nginx:alpine AS production

# Copy custom nginx config
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Copy built app from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

# Development stage (for development with hot reload)
FROM node:18-alpine AS development

WORKDIR /app

# Install global dependencies
RUN npm install -g @expo/cli

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Expose ports
EXPOSE 19006 3000

# Start development server
CMD ["npm", "start"]
