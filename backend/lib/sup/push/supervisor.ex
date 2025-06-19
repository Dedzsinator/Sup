defmodule Sup.Push.Supervisor do
  @moduledoc """
  Supervisor for push notification-related processes.
  """

  use Supervisor

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    children = [
      Sup.Push.PushService
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
