defmodule Sup.Auth.Supervisor do
  @moduledoc """
  Supervisor for authentication-related processes.
  """

  use Supervisor

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    children = [
      # Add any auth-related GenServers here
      # For now, auth is stateless
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
