defmodule Sup.Security.MessageExpiry do
  @moduledoc """
  Simplified message expiry and auto-deletion system for enhanced privacy.
  """

  use GenServer
  require Logger
  alias Sup.Repo
  alias Sup.Messaging.Message
  import Ecto.Query

  # Run cleanup every 5 minutes
  @cleanup_interval :timer.minutes(5)
  # 7 days default
  @default_expiry_hours 24 * 7

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Set message expiry time
  """
  def set_message_expiry(message_id, expiry_type, expiry_value) do
    GenServer.cast(__MODULE__, {:set_expiry, message_id, expiry_type, expiry_value})
  end

  @doc """
  Force cleanup of expired messages
  """
  def force_cleanup do
    GenServer.cast(__MODULE__, :force_cleanup)
  end

  # GenServer Callbacks

  @impl true
  def init(_opts) do
    schedule_cleanup()
    {:ok, %{}}
  end

  @impl true
  def handle_cast({:set_expiry, message_id, expiry_type, expiry_value}, state) do
    set_expiry_metadata(message_id, expiry_type, expiry_value)
    {:noreply, state}
  end

  @impl true
  def handle_cast(:force_cleanup, state) do
    cleanup_expired_messages()
    {:noreply, state}
  end

  @impl true
  def handle_info(:cleanup, state) do
    cleanup_expired_messages()
    schedule_cleanup()
    {:noreply, state}
  end

  # Private Functions

  defp schedule_cleanup do
    Process.send_after(self(), :cleanup, @cleanup_interval)
  end

  defp set_expiry_metadata(message_id, expiry_type, expiry_value) do
    expiry_time = calculate_expiry_time(expiry_type, expiry_value)

    query = from(m in Message, where: m.id == ^message_id)
    Repo.update_all(query, set: [expires_at: expiry_time])
  end

  defp calculate_expiry_time(expiry_type, expiry_value) do
    case expiry_type do
      "timer" ->
        DateTime.utc_now()
        # expiry_value in minutes
        |> DateTime.add(expiry_value * 60, :second)

      "daily" ->
        # Expire at end of day
        DateTime.utc_now()
        |> DateTime.add(24 * 60 * 60, :second)

      "weekly" ->
        # Expire in 7 days
        DateTime.utc_now()
        |> DateTime.add(7 * 24 * 60 * 60, :second)

      _ ->
        # Default expiry
        DateTime.utc_now()
        |> DateTime.add(@default_expiry_hours * 60 * 60, :second)
    end
  end

  defp cleanup_expired_messages do
    now = DateTime.utc_now()

    # Find all expired messages
    expired_query =
      from(m in Message,
        where: not is_nil(m.expires_at) and m.expires_at <= ^now
      )

    expired_messages = Repo.all(expired_query)

    if length(expired_messages) > 0 do
      Logger.info("Cleaning up #{length(expired_messages)} expired messages")

      # Delete expired messages
      case Application.get_env(:sup, :message_deletion_type, :soft) do
        :soft ->
          Repo.update_all(expired_query, set: [deleted_at: DateTime.utc_now()])

        :hard ->
          Repo.delete_all(expired_query)
      end

      # Log cleanup activity
      Sup.Security.AuditLog.log_security_violation(
        "message_cleanup",
        %{
          cleaned_count: length(expired_messages),
          cleanup_time: now
        },
        %{},
        "low"
      )
    end
  end
end
