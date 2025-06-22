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

# Security configuration
config :sup,
  # Rate limiting with Hammer
  rate_limiting_enabled: true,

  # Message expiry settings
  # :soft or :hard
  message_deletion_type: :soft,
  # 7 days
  default_message_expiry_hours: 24 * 7,

  # Security monitoring
  security_monitoring_enabled: true,
  suspicious_activity_threshold: 10,

  # Two-factor authentication
  tfa_issuer: "Sup Messaging App",
  tfa_digits: 6,
  tfa_period: 30,

  # Encryption settings
  encryption_algorithm: :aes_256_gcm,
  key_derivation_iterations: 100_000,

  # Account security
  max_failed_login_attempts: 5,
  account_lockout_duration_minutes: 30,

  # Session management
  session_timeout_hours: 24,
  refresh_token_ttl_days: 30

# Hammer rate limiting configuration
config :hammer,
  backend: {Hammer.Backend.ETS, [expiry_ms: 60_000 * 60 * 4, cleanup_interval_ms: 60_000 * 10]}

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
