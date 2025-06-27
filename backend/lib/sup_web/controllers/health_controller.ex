defmodule SupWeb.HealthController do
  use Phoenix.Controller

  def health(conn, _params) do
    json(conn, %{
      status: "ok",
      timestamp: DateTime.utc_now(),
      version: Application.spec(:sup, :vsn) || "unknown"
    })
  end
end
