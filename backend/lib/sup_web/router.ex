defmodule SupWeb.Router do
  use Phoenix.Router

  pipeline :api do
    plug(:accepts, ["json"])
    plug(:fetch_session)
    plug(:protect_from_forgery)
    plug(:put_secure_browser_headers)
  end

  scope "/api", SupWeb do
    pipe_through(:api)

    # Health check
    get("/health", HealthController, :health)

    # API routes are handled by the Sup.ApiRouter
    forward("/", Sup.ApiRouter)
  end

  # Enables LiveDashboard only for development
  if Mix.env() in [:dev, :test] do
    import Phoenix.LiveDashboard.Router

    scope "/" do
      pipe_through([:fetch_session, :protect_from_forgery])

      live_dashboard("/dashboard", metrics: SupWeb.Telemetry)
    end
  end
end
