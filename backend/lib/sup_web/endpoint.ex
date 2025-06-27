defmodule SupWeb.Endpoint do
  @moduledoc """
  Phoenix endpoint for the Sup application.
  """

  use Phoenix.Endpoint, otp_app: :sup

  # The session will be stored in the cookie and signed,
  # this means its contents can be read but not tampered with.
  # Set :encryption_salt if you would also like to encrypt it.
  @session_options [
    store: :cookie,
    key: "_sup_key",
    signing_salt: "sup_signing_salt"
  ]

  socket("/live", Phoenix.LiveView.Socket, websocket: [connect_info: [session: @session_options]])

  # Serve at "/" the static files from "priv/static" directory.
  plug(Plug.Static,
    at: "/",
    from: :sup,
    gzip: false,
    only: ~w(assets fonts images favicon.ico robots.txt)
  )

  # Code reloading can be explicitly enabled under the
  # :code_reloader configuration of your endpoint.
  if code_reloading? do
    socket("/phoenix/live_reload/socket", Phoenix.LiveReloader.Socket)
    plug(Phoenix.LiveReloader)
    plug(Phoenix.CodeReloader)
    plug(Phoenix.Ecto.CheckRepoStatus, otp_app: :sup)
  end

  plug(Phoenix.LiveDashboard.RequestLogger,
    param_key: "request_logger",
    cookie_key: "request_logger"
  )

  plug(Plug.RequestId)
  plug(Plug.Telemetry, event_prefix: [:phoenix, :endpoint])

  plug(Plug.Parsers,
    parsers: [:urlencoded, :multipart, :json],
    pass: ["*/*"],
    json_decoder: Phoenix.json_library()
  )

  plug(Plug.MethodOverride)
  plug(Plug.Head)
  plug(Plug.Session, @session_options)
  plug(SupWeb.Router)

  @doc """
  Broadcast a message to a topic
  """
  def broadcast(topic, event, payload) do
    Phoenix.PubSub.broadcast(Sup.PubSub, topic, {event, payload})
  end

  @doc """
  Broadcast a message to a topic from a specific process
  """
  def broadcast_from(from_pid, topic, event, payload) do
    Phoenix.PubSub.broadcast_from(Sup.PubSub, from_pid, topic, {event, payload})
  end

  @doc """
  Subscribe to a topic
  """
  def subscribe(topic) do
    Phoenix.PubSub.subscribe(Sup.PubSub, topic)
  end

  @doc """
  Unsubscribe from a topic
  """
  def unsubscribe(topic) do
    Phoenix.PubSub.unsubscribe(Sup.PubSub, topic)
  end
end
