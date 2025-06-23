defmodule Sup.Messaging.AnalyticsService do
  @moduledoc """
  Service for collecting and analyzing messaging analytics.
  Tracks message patterns, user engagement, and room activity.
  """

  use GenServer
  require Logger

  alias Sup.Messaging.{Message, MessageAnalytics}
  alias Sup.Room.Room
  alias Sup.Repo
  import Ecto.Query

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Track a message event
  """
  def track_message(user_id, room_id, event_type, metadata \\ %{}) do
    GenServer.cast(__MODULE__, {:track_message, user_id, room_id, event_type, metadata})
  end

  @doc """
  Track user activity
  """
  def track_activity(user_id, activity_type, metadata \\ %{}) do
    GenServer.cast(__MODULE__, {:track_activity, user_id, activity_type, metadata})
  end

  @doc """
  Get message analytics for a room
  """
  def get_room_analytics(room_id, start_date, end_date) do
    GenServer.call(__MODULE__, {:get_room_analytics, room_id, start_date, end_date})
  end

  @doc """
  Get user activity analytics
  """
  def get_user_analytics(user_id, start_date, end_date) do
    GenServer.call(__MODULE__, {:get_user_analytics, user_id, start_date, end_date})
  end

  @doc """
  Get global analytics
  """
  def get_global_analytics(start_date, end_date) do
    GenServer.call(__MODULE__, {:get_global_analytics, start_date, end_date})
  end

  # GenServer callbacks

  @impl true
  def init(_opts) do
    Logger.info("Analytics Service started")
    {:ok, %{}}
  end

  @impl true
  def handle_cast({:track_message, user_id, room_id, event_type, metadata}, state) do
    spawn(fn ->
      analytics_data = %{
        user_id: user_id,
        room_id: room_id,
        event_type: event_type,
        metadata: metadata,
        timestamp: DateTime.utc_now()
      }

      case MessageAnalytics.changeset(%MessageAnalytics{}, analytics_data) |> Repo.insert() do
        {:ok, _} ->
          Logger.debug("Message analytics tracked: #{event_type} for user #{user_id}")

        {:error, changeset} ->
          Logger.error("Failed to track message analytics: #{inspect(changeset.errors)}")
      end
    end)

    {:noreply, state}
  end

  @impl true
  def handle_cast({:track_activity, user_id, activity_type, metadata}, state) do
    spawn(fn ->
      # Store user activity in a separate table or Redis for fast access
      activity_key = "user_activity:#{user_id}:#{Date.utc_today()}"
      
      case Redix.command(:redix, ["HINCRBY", activity_key, activity_type, 1]) do
        {:ok, _} ->
          # Set expiration to 30 days
          Redix.command(:redix, ["EXPIRE", activity_key, 2_592_000])
          Logger.debug("Activity tracked: #{activity_type} for user #{user_id}")

        {:error, error} ->
          Logger.error("Failed to track activity: #{inspect(error)}")
      end
    end)

    {:noreply, state}
  end

  @impl true
  def handle_call({:get_room_analytics, room_id, start_date, end_date}, _from, state) do
    analytics = get_room_analytics_data(room_id, start_date, end_date)
    {:reply, {:ok, analytics}, state}
  end

  @impl true
  def handle_call({:get_user_analytics, user_id, start_date, end_date}, _from, state) do
    analytics = get_user_analytics_data(user_id, start_date, end_date)
    {:reply, {:ok, analytics}, state}
  end

  @impl true
  def handle_call({:get_global_analytics, start_date, end_date}, _from, state) do
    analytics = get_global_analytics_data(start_date, end_date)
    {:reply, {:ok, analytics}, state}
  end

  # Private functions

  defp get_room_analytics_data(room_id, start_date, end_date) do
    message_count = from(m in Message,
      where: m.room_id == ^room_id and
             m.inserted_at >= ^start_date and
             m.inserted_at <= ^end_date,
      select: count(m.id)
    ) |> Repo.one()

    active_users = from(m in Message,
      where: m.room_id == ^room_id and
             m.inserted_at >= ^start_date and
             m.inserted_at <= ^end_date,
      select: m.sender_id,
      distinct: true
    ) |> Repo.all() |> length()

    hourly_activity = from(m in Message,
      where: m.room_id == ^room_id and
             m.inserted_at >= ^start_date and
             m.inserted_at <= ^end_date,
      group_by: fragment("date_part('hour', ?)", m.inserted_at),
      select: {fragment("date_part('hour', ?)", m.inserted_at), count(m.id)},
      order_by: fragment("date_part('hour', ?)", m.inserted_at)
    ) |> Repo.all()

    %{
      room_id: room_id,
      message_count: message_count,
      active_users: active_users,
      hourly_activity: hourly_activity,
      start_date: start_date,
      end_date: end_date
    }
  end

  defp get_user_analytics_data(user_id, start_date, end_date) do
    message_count = from(m in Message,
      where: m.sender_id == ^user_id and
             m.inserted_at >= ^start_date and
             m.inserted_at <= ^end_date,
      select: count(m.id)
    ) |> Repo.one()

    active_rooms = from(m in Message,
      where: m.sender_id == ^user_id and
             m.inserted_at >= ^start_date and
             m.inserted_at <= ^end_date,
      select: m.room_id,
      distinct: true
    ) |> Repo.all() |> length()

    daily_activity = from(m in Message,
      where: m.sender_id == ^user_id and
             m.inserted_at >= ^start_date and
             m.inserted_at <= ^end_date,
      group_by: fragment("date(?)", m.inserted_at),
      select: {fragment("date(?)", m.inserted_at), count(m.id)},
      order_by: fragment("date(?)", m.inserted_at)
    ) |> Repo.all()

    %{
      user_id: user_id,
      message_count: message_count,
      active_rooms: active_rooms,
      daily_activity: daily_activity,
      start_date: start_date,
      end_date: end_date
    }
  end

  defp get_global_analytics_data(start_date, end_date) do
    total_messages = from(m in Message,
      where: m.inserted_at >= ^start_date and m.inserted_at <= ^end_date,
      select: count(m.id)
    ) |> Repo.one()

    active_users = from(m in Message,
      where: m.inserted_at >= ^start_date and m.inserted_at <= ^end_date,
      select: m.sender_id,
      distinct: true
    ) |> Repo.all() |> length()

    active_rooms = from(m in Message,
      where: m.inserted_at >= ^start_date and m.inserted_at <= ^end_date,
      select: m.room_id,
      distinct: true
    ) |> Repo.all() |> length()

    peak_hours = from(m in Message,
      where: m.inserted_at >= ^start_date and m.inserted_at <= ^end_date,
      group_by: fragment("date_part('hour', ?)", m.inserted_at),
      select: {fragment("date_part('hour', ?)", m.inserted_at), count(m.id)},
      order_by: [desc: count(m.id)],
      limit: 5
    ) |> Repo.all()

    %{
      total_messages: total_messages,
      active_users: active_users,
      active_rooms: active_rooms,
      peak_hours: peak_hours,
      start_date: start_date,
      end_date: end_date
    }
  end
end
