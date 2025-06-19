# Configuration for the Sup application
import Config

# Configure Ecto repositories
config :sup, ecto_repos: [Sup.Repo]

# Database configuration
config :sup, Sup.Repo,
  username: System.get_env("DATABASE_USER") || "postgres",
  password: System.get_env("DATABASE_PASSWORD") || "postgres",
  hostname: System.get_env("DATABASE_HOST") || "localhost",
  database: System.get_env("DATABASE_NAME") || "sup_dev",
  port: String.to_integer(System.get_env("DATABASE_PORT") || "5432"),
  pool_size: String.to_integer(System.get_env("DATABASE_POOL_SIZE") || "10"),
  show_sensitive_data_on_connection_error: true

# Redis configuration (disabled)
# config :sup,
#   redis_url: System.get_env("REDIS_URL") || "redis://localhost:6379"

# ScyllaDB configuration
config :sup,
  scylla_nodes: String.split(System.get_env("SCYLLA_NODES") || "127.0.0.1:9042", ","),
  scylla_keyspace: System.get_env("SCYLLA_KEYSPACE") || "sup",
  autocomplete_service_url: System.get_env("AUTOCOMPLETE_SERVICE_URL") || "http://localhost:8000"

# Guardian configuration for JWT
config :sup, Sup.Auth.Guardian,
  issuer: "sup",
  secret_key: System.get_env("GUARDIAN_SECRET_KEY") || "your-secret-key-here-change-in-production"

# Push notification configuration
config :sup,
  fcm_server_key: System.get_env("FCM_SERVER_KEY"),
  apns_cert_path: System.get_env("APNS_CERT_PATH"),
  web_push_keys: %{
    public_key: System.get_env("WEB_PUSH_PUBLIC_KEY"),
    private_key: System.get_env("WEB_PUSH_PRIVATE_KEY")
  }

# Clustering configuration (disabled for development)
# config :libcluster,
#   topologies: [
#     sup: [
#       strategy: Cluster.Strategy.Epmd,
#       config: [hosts: []]
#     ]
#   ]

# Logging configuration
config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id, :user_id, :room_id]

# Import environment specific config
import_config "#{config_env()}.exs"
