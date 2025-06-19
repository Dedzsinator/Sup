# Use Node.js official image
FROM node:18-alpine

# Set working directory
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

# Start the development server
CMD ["npm", "start"]
