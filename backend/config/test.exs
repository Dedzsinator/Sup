import Config

# Test configuration
config :sup, Sup.Repo,
  username: "postgres",
  password: "postgres",
  hostname: "localhost",
  database: "sup_test#{System.get_env("MIX_TEST_PARTITION")}",
  pool: Ecto.Adapters.SQL.Sandbox,
  pool_size: 10

# Print only warnings and errors during test
config :logger, level: :warn

# Reduce bcrypt rounds for faster tests
config :argon2_elixir, t_cost: 1, m_cost: 8

# Guardian test configuration
config :sup, Sup.Auth.Guardian, secret_key: "test-secret-key"
