# Production Multi-stage Dockerfile for Elixir Backend
FROM elixir:1.15-alpine AS deps

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

# Set build ENV
ENV MIX_ENV=prod

# Install dependencies
RUN mix deps.get --only=prod && \
    mix deps.compile

# Build stage
FROM deps AS build

# Copy source code
COPY . .

# Compile the project
RUN mix compile

# Create release
RUN mix release

# Runtime stage
FROM alpine:3.18 AS runtime

# Install runtime dependencies
RUN apk add --no-cache \
    openssl \
    ncurses-libs \
    libstdc++

# Create app user
RUN addgroup -g 1001 -S app && \
    adduser -S -D -H -u 1001 -h /app -s /bin/sh -G app app

# Set working directory
WORKDIR /app

# Copy release from build stage
COPY --from=build --chown=app:app /app/_build/prod/rel/sup ./

# Switch to app user
USER app

# Expose port
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/bin/sup eval "Sup.HealthCheck.check()" || exit 1

# Start the application
CMD ["/app/bin/sup", "start"]
