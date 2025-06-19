defmodule Sup.Messaging.QueueProcessor do
  @moduledoc """
  GenServer for processing message queues and handling backpressure.
  """

  use GenServer
  require Logger

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    # Schedule periodic processing
    Process.send_after(self(), :process_queue, 1000)

    {:ok,
     %{
       queue: :queue.new(),
       processing: false
     }}
  end

  @impl true
  def handle_info(:process_queue, state) do
    new_state = process_messages(state)

    # Schedule next processing
    Process.send_after(self(), :process_queue, 1000)

    {:noreply, new_state}
  end

  def handle_cast({:queue_message, message}, state) do
    new_queue = :queue.in(message, state.queue)
    {:noreply, %{state | queue: new_queue}}
  end

  defp process_messages(%{queue: queue, processing: true} = state) do
    # Already processing, skip
    state
  end

  defp process_messages(%{queue: queue} = state) do
    case :queue.out(queue) do
      {{:value, message}, new_queue} ->
        # Process message asynchronously
        Task.Supervisor.start_child(Sup.Messaging.TaskSupervisor, fn ->
          process_single_message(message)
        end)

        process_messages(%{state | queue: new_queue, processing: true})

      {:empty, _queue} ->
        %{state | processing: false}
    end
  end

  defp process_single_message(message) do
    # Handle different message types
    case message.type do
      :delivery_receipt ->
        handle_delivery_receipt(message)

      :push_notification ->
        handle_push_notification(message)

      _ ->
        Logger.warn("Unknown message type: #{message.type}")
    end
  end

  defp handle_delivery_receipt(_message) do
    # Update delivery status
    :ok
  end

  defp handle_push_notification(_message) do
    # Send push notification
    :ok
  end
end
