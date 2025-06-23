defmodule Sup.Application do
  @moduledoc """
  The Sup Application module - Main OTP application supervisor.
  """

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Database & Storage
      Sup.Repo,
      # Sup.Redis, # Disabled - not needed for development
      Sup.ScyllaDB,

      # PubSub for real-time messaging
      {Phoenix.PubSub, name: Sup.PubSub},

      # Security Services
      Sup.Security.Monitor,
      Sup.Security.MessageExpiry,

      # Enhanced Services
      Sup.Messaging.OfflineQueueService,
      Sup.Analytics.AnalyticsService,
      Sup.Sync.MultiDeviceSyncService,

      # Core business logic supervisors
      Sup.Auth.Supervisor,
      Sup.Messaging.Supervisor,
      Sup.Presence.Supervisor,
      Sup.Room.Supervisor,
      Sup.Push.Supervisor,

      # WebSocket connection registry
      {Registry, keys: :unique, name: Sup.ConnectionRegistry},

      # HTTP Server (Plug + Cowboy)
      {Plug.Cowboy, scheme: :http, plug: Sup.Router, options: [port: 4000]},

      # WebSocket Server
      {Plug.Cowboy.Drainer, refs: [Sup.Router]}
    ]

    # Clustering support for horizontal scaling (disabled for development)
    # topologies = Application.get_env(:libcluster, :topologies) || []
    # children = [{Cluster.Supervisor, [topologies, [name: Sup.ClusterSupervisor]]} | children]

    opts = [strategy: :one_for_one, name: Sup.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
