import Config

# Development-specific configuration
config :sup, Sup.Repo,
  database: "sup_dev",
  hostname: "localhost",
  show_sensitive_data_on_connection_error: true,
  pool_size: 10

# Enable code reloading for development
config :sup, dev_routes: true

# Do not include metadata nor timestamps in development logs
config :logger, :console, format: "[$level] $message\n"

# Set a higher stacktrace during development
config :logger, level: :debug

# Guardian secret key for development (change in production)
config :sup, Sup.Auth.Guardian, secret_key: "dev-secret-key-change-in-production"

# Spam Detection Service Configuration
config :sup,
  spam_detection_url: "http://localhost:8082",
  spam_detection_api_key: "development-spam-api-key"
