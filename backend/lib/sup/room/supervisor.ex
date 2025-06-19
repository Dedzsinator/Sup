defmodule Sup.Room.Supervisor do
  @moduledoc """
  Supervisor for room-related processes.
  """

  use Supervisor

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    children = [
      # Room management processes can be added here
      # For now, room service is stateless
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
