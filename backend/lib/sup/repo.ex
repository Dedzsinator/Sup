defmodule Sup.Repo do
  @moduledoc """
  PostgreSQL repository for structured data (users, rooms, metadata).
  """

  use Ecto.Repo,
    otp_app: :sup,
    adapter: Ecto.Adapters.Postgres
end
