defmodule Sup.Messaging.OfflineQueueService do
  @moduledoc """
  Service for managing offline message queues.
  Stores messages for offline users and delivers them when they come online.
  """

  use GenServer
  require Logger

  alias Sup.Messaging.OfflineMessage
  alias Sup.Presence.EnhancedPresenceService
  alias Sup.Repo
  import Ecto.Query

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Queue a message for offline delivery
  """
  def queue_message(user_id, message_data) do
    GenServer.cast(__MODULE__, {:queue_message, user_id, message_data})
  end

  @doc """
  Process queued messages for a user (when they come online)
  """
  def process_user_queue(user_id) do
    GenServer.cast(__MODULE__, {:process_user_queue, user_id})
  end

  @doc """
  Get pending message count for a user
  """
  def get_pending_count(user_id) do
    GenServer.call(__MODULE__, {:get_pending_count, user_id})
  end

  @doc """
  Clear all queued messages for a user
  """
  def clear_user_queue(user_id) do
    GenServer.cast(__MODULE__, {:clear_user_queue, user_id})
  end

  # GenServer callbacks

  @impl true
  def init(_opts) do
    # Schedule periodic cleanup of old messages
    schedule_cleanup()
    {:ok, %{}}
  end

  @impl true
  def handle_cast({:queue_message, user_id, message_data}, state) do
    # Check if user is actually offline
    case EnhancedPresenceService.get_user_presence(user_id) do
      {:ok, %{status: :offline}} ->
        queue_offline_message(user_id, message_data)

      {:ok, %{status: :invisible}} ->
        queue_offline_message(user_id, message_data)

      _ ->
        # User is online, no need to queue
        :ok
    end

    {:noreply, state}
  end

  @impl true
  def handle_cast({:process_user_queue, user_id}, state) do
    deliver_queued_messages(user_id)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:clear_user_queue, user_id}, state) do
    from(m in OfflineMessage, where: m.user_id == ^user_id)
    |> Repo.delete_all()

    {:noreply, state}
  end

  @impl true
  def handle_call({:get_pending_count, user_id}, _from, state) do
    count = from(m in OfflineMessage, where: m.user_id == ^user_id)
            |> Repo.aggregate(:count, :id)

    {:reply, count, state}
  end

  @impl true
  def handle_info(:cleanup_old_messages, state) do
    cleanup_old_messages()
    schedule_cleanup()
    {:reply, state}
  end

  # Private functions

  defp queue_offline_message(user_id, message_data) do
    offline_message_attrs = %{
      user_id: user_id,
      message_type: message_data[:type] || "message",
      message_id: message_data[:id],
      room_id: message_data[:room_id],
      sender_id: message_data[:sender_id],
      content: message_data[:content],
      metadata: message_data[:metadata] || %{},
      priority: determine_priority(message_data),
      expires_at: DateTime.utc_now() |> DateTime.add(7 * 24 * 3600, :second) # 7 days
    }

    case OfflineMessage.changeset(%OfflineMessage{}, offline_message_attrs) |> Repo.insert() do
      {:ok, _offline_message} ->
        Logger.info("Queued offline message for user #{user_id}")
        :ok

      {:error, changeset} ->
        Logger.error("Failed to queue offline message: #{inspect(changeset)}")
        {:error, changeset}
    end
  end

  defp deliver_queued_messages(user_id) do
    queued_messages = from(m in OfflineMessage,
                          where: m.user_id == ^user_id,
                          order_by: [asc: m.priority, asc: m.inserted_at])
                     |> Repo.all()

    if length(queued_messages) > 0 do
      Logger.info("Delivering #{length(queued_messages)} queued messages to user #{user_id}")

      # Group messages by type for efficient delivery
      grouped_messages = Enum.group_by(queued_messages, & &1.message_type)

      # Deliver each group
      Enum.each(grouped_messages, fn {message_type, messages} ->
        deliver_message_group(user_id, message_type, messages)
      end)

      # Clean up delivered messages
      message_ids = Enum.map(queued_messages, & &1.id)
      from(m in OfflineMessage, where: m.id in ^message_ids)
      |> Repo.delete_all()
    end
  end

  defp deliver_message_group(user_id, "message", messages) do
    # Deliver regular messages
    Enum.each(messages, fn offline_msg ->
      message_data = %{
        id: offline_msg.message_id,
        room_id: offline_msg.room_id,
        sender_id: offline_msg.sender_id,
        content: offline_msg.content,
        type: "message",
        timestamp: offline_msg.inserted_at,
        metadata: offline_msg.metadata,
        offline_queued: true
      }

      Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{user_id}", {:offline_message, message_data})
    end)
  end

  defp deliver_message_group(user_id, "reaction", messages) do
    # Deliver reaction notifications
    Enum.each(messages, fn offline_msg ->
      Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{user_id}", {:offline_reaction, offline_msg.metadata})
    end)
  end

  defp deliver_message_group(user_id, "mention", messages) do
    # Deliver mention notifications
    Enum.each(messages, fn offline_msg ->
      Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{user_id}", {:offline_mention, offline_msg.metadata})
    end)
  end

  defp deliver_message_group(user_id, "call", messages) do
    # Deliver missed call notifications
    Enum.each(messages, fn offline_msg ->
      Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{user_id}", {:missed_call, offline_msg.metadata})
    end)
  end

  defp deliver_message_group(user_id, _message_type, messages) do
    # Deliver other message types generically
    Enum.each(messages, fn offline_msg ->
      Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{user_id}", {:offline_notification, offline_msg})
    end)
  end

  defp determine_priority(message_data) do
    case message_data do
      %{type: "call"} -> 1  # Highest priority
      %{metadata: %{mentions: mentions}} when length(mentions) > 0 -> 2  # High priority for mentions
      %{type: "direct_message"} -> 3  # Medium-high priority for DMs
      _ -> 4  # Normal priority
    end
  end

  defp cleanup_old_messages do
    cutoff_time = DateTime.utc_now() |> DateTime.add(-7 * 24 * 3600, :second)

    {deleted_count, _} = from(m in OfflineMessage, where: m.expires_at < ^cutoff_time)
                        |> Repo.delete_all()

    if deleted_count > 0 do
      Logger.info("Cleaned up #{deleted_count} expired offline messages")
    end
  end

  defp schedule_cleanup do
    # Schedule cleanup every hour
    Process.send_after(self(), :cleanup_old_messages, :timer.hours(1))
  end
end
