import Config

# Production configuration
config :sup, Sup.Repo,
  url: System.get_env("DATABASE_URL"),
  pool_size: String.to_integer(System.get_env("POOL_SIZE") || "10"),
  ssl: true

# Force SSL connections in production
config :sup, force_ssl: true

# Configure Guardian with environment variables
config :sup, Sup.Auth.Guardian, secret_key: System.fetch_env!("GUARDIAN_SECRET_KEY")

# Configure logging for production
config :logger, level: :info

# Runtime production config
if System.get_env("PHX_SERVER") do
  config :sup, start_server: true
end
