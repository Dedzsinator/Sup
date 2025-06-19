defmodule Sup.Messaging.Supervisor do
  @moduledoc """
  Supervisor for messaging-related processes.
  """

  use Supervisor

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    children = [
      # Message processing workers
      {Task.Supervisor, name: Sup.Messaging.TaskSupervisor},

      # Message queue processor (for batch operations)
      Sup.Messaging.QueueProcessor
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
