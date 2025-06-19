# Use official Elixir image
FROM elixir:1.15-alpine

# Set environment variables
ENV MIX_ENV=prod

# Install system dependencies
RUN apk add --no-cache \
    build-base \
    git \
    nodejs \
    npm

# Create app directory
WORKDIR /app

# Install Hex and Rebar
RUN mix local.hex --force && \
    mix local.rebar --force

# Copy mix files
COPY mix.exs mix.lock ./

# Install dependencies
RUN mix deps.get && mix deps.compile

# Copy source code
COPY . .

# Compile the project
RUN mix compile

# Create release
RUN mix release

# Expose port
EXPOSE 4000

# Start the application
CMD ["_build/prod/rel/sup/bin/sup", "start"]
