defmodule Sup.MixProject do
  use Mix.Project

  def project do
    [
      app: :sup,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.detail": :test,
        "coveralls.post": :test,
        "coveralls.html": :test
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger, :crypto, :ssl],
      mod: {Sup.Application, []}
    ]
  end

  defp deps do
    [
      # Web server
      {:plug, "~> 1.14"},
      {:plug_cowboy, "~> 2.6"},
      {:cors_plug, "~> 3.0"},

      # Database & Storage
      {:ecto_sql, "~> 3.10"},
      {:postgrex, "~> 0.17"},
      {:redix, "~> 1.2"},
      # ScyllaDB driver
      {:xandra, "~> 0.14"},

      # Real-time & WebSockets
      {:phoenix_pubsub, "~> 2.1"},
      {:websock_adapter, "~> 0.5"},

      # Authentication & Security
      {:argon2_elixir, "~> 3.1"},
      {:guardian, "~> 2.3"},
      {:comeonin, "~> 5.3"},
      {:hammer, "~> 6.0"},
      {:eqrcode, "~> 0.1.10"},
      {:pot, "~> 1.0"},
      {:ex_audit, "~> 0.9"},
      {:secure_random, "~> 0.5"},
      {:jose, "~> 1.11"},
      {:telemetry, "~> 1.2"},
      {:telemetry_metrics, "~> 0.6"},
      {:telemetry_poller, "~> 1.0"},

      # JSON & Data processing
      {:jason, "~> 1.4"},
      {:ecto_enum, "~> 1.4"},
      {:httpoison, "~> 2.0"},

      # AI & Vector search
      {:nx, "~> 0.6"},
      {:scholar, "~> 0.2"},

      # Development & Testing
      {:mix_test_watch, "~> 1.1", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.3", only: [:dev], runtime: false},
      {:ex_machina, "~> 2.7", only: :test},
      {:bypass, "~> 2.1", only: :test},
      {:mox, "~> 1.0", only: :test},
      {:excoveralls, "~> 0.17", only: :test}
    ]
  end

  defp aliases do
    [
      "ecto.setup": ["ecto.create", "ecto.migrate", "run priv/repo/seeds.exs"],
      "ecto.reset": ["ecto.drop", "ecto.setup"],
      test: ["ecto.create --quiet", "ecto.migrate --quiet", "test"]
    ]
  end
end
