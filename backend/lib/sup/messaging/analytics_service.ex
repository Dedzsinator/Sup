defmodule Sup.Messaging.AnalyticsService do
  @moduledoc """
  Service for collecting and analyzing messaging analytics.
  Tracks message patterns, user engagement, and room activity.
  """

  use GenServer
  require Logger

  alias Sup.Messaging.{Message, MessageAnalytics}
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

  @doc """
  Get message analytics for specific criteria
  """
  def get_message_analytics(user_id, room_id, date_range) do
    %{start_date: start_date, end_date: end_date} = date_range

    query =
      from(a in MessageAnalytics,
        where: a.timestamp >= ^start_date and a.timestamp <= ^end_date
      )

    query = if user_id, do: where(query, [a], a.user_id == ^user_id), else: query
    query = if room_id, do: where(query, [a], a.room_id == ^room_id), else: query

    analytics = Repo.all(query)

    summary = %{
      total_messages: length(analytics),
      date_range: date_range,
      user_id: user_id,
      room_id: room_id,
      breakdown: group_analytics_by_date(analytics)
    }

    {:ok, summary}
  end

  @doc """
  Get room insights including activity patterns and user engagement
  """
  def get_room_insights(room_id) do
    end_date = DateTime.utc_now()
    # Last 30 days
    start_date = DateTime.add(end_date, -30, :day)

    insights = %{
      room_id: room_id,
      period: "30_days",
      message_count: get_room_message_count(room_id, start_date, end_date),
      active_users: get_room_active_users(room_id, start_date, end_date),
      peak_activity_hours: get_room_peak_hours(room_id, start_date, end_date),
      engagement_score: calculate_room_engagement(room_id, start_date, end_date),
      growth_trend: calculate_room_growth_trend(room_id)
    }

    {:ok, insights}
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
  def handle_cast({:track_activity, user_id, activity_type, _metadata}, state) do
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
    message_count =
      from(m in Message,
        where:
          m.room_id == ^room_id and
            m.inserted_at >= ^start_date and
            m.inserted_at <= ^end_date,
        select: count(m.id)
      )
      |> Repo.one()

    active_users =
      from(m in Message,
        where:
          m.room_id == ^room_id and
            m.inserted_at >= ^start_date and
            m.inserted_at <= ^end_date,
        select: m.sender_id,
        distinct: true
      )
      |> Repo.all()
      |> length()

    hourly_activity =
      from(m in Message,
        where:
          m.room_id == ^room_id and
            m.inserted_at >= ^start_date and
            m.inserted_at <= ^end_date,
        group_by: fragment("date_part('hour', ?)", m.inserted_at),
        select: {fragment("date_part('hour', ?)", m.inserted_at), count(m.id)},
        order_by: fragment("date_part('hour', ?)", m.inserted_at)
      )
      |> Repo.all()

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
    message_count =
      from(m in Message,
        where:
          m.sender_id == ^user_id and
            m.inserted_at >= ^start_date and
            m.inserted_at <= ^end_date,
        select: count(m.id)
      )
      |> Repo.one()

    active_rooms =
      from(m in Message,
        where:
          m.sender_id == ^user_id and
            m.inserted_at >= ^start_date and
            m.inserted_at <= ^end_date,
        select: m.room_id,
        distinct: true
      )
      |> Repo.all()
      |> length()

    daily_activity =
      from(m in Message,
        where:
          m.sender_id == ^user_id and
            m.inserted_at >= ^start_date and
            m.inserted_at <= ^end_date,
        group_by: fragment("date(?)", m.inserted_at),
        select: {fragment("date(?)", m.inserted_at), count(m.id)},
        order_by: fragment("date(?)", m.inserted_at)
      )
      |> Repo.all()

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
    total_messages =
      from(m in Message,
        where: m.inserted_at >= ^start_date and m.inserted_at <= ^end_date,
        select: count(m.id)
      )
      |> Repo.one()

    active_users =
      from(m in Message,
        where: m.inserted_at >= ^start_date and m.inserted_at <= ^end_date,
        select: m.sender_id,
        distinct: true
      )
      |> Repo.all()
      |> length()

    active_rooms =
      from(m in Message,
        where: m.inserted_at >= ^start_date and m.inserted_at <= ^end_date,
        select: m.room_id,
        distinct: true
      )
      |> Repo.all()
      |> length()

    peak_hours =
      from(m in Message,
        where: m.inserted_at >= ^start_date and m.inserted_at <= ^end_date,
        group_by: fragment("date_part('hour', ?)", m.inserted_at),
        select: {fragment("date_part('hour', ?)", m.inserted_at), count(m.id)},
        order_by: [desc: count(m.id)],
        limit: 5
      )
      |> Repo.all()

    %{
      total_messages: total_messages,
      active_users: active_users,
      active_rooms: active_rooms,
      peak_hours: peak_hours,
      start_date: start_date,
      end_date: end_date
    }
  end

  defp group_analytics_by_date(analytics) do
    analytics
    |> Enum.group_by(fn record -> Date.to_string(DateTime.to_date(record.timestamp)) end)
    |> Enum.map(fn {date, records} ->
      %{
        date: date,
        message_count: length(records),
        unique_users: records |> Enum.map(& &1.user_id) |> Enum.uniq() |> length()
      }
    end)
    |> Enum.sort_by(& &1.date)
  end

  defp get_room_message_count(room_id, start_date, end_date) do
    from(a in MessageAnalytics,
      where:
        a.room_id == ^room_id and
          a.timestamp >= ^start_date and
          a.timestamp <= ^end_date,
      select: count(a.id)
    )
    |> Repo.one() || 0
  end

  defp get_room_active_users(room_id, start_date, end_date) do
    from(a in MessageAnalytics,
      where:
        a.room_id == ^room_id and
          a.timestamp >= ^start_date and
          a.timestamp <= ^end_date,
      distinct: a.user_id,
      select: count(a.user_id)
    )
    |> Repo.one() || 0
  end

  defp get_room_peak_hours(room_id, start_date, end_date) do
    # Get message counts by hour of day
    from(a in MessageAnalytics,
      where:
        a.room_id == ^room_id and
          a.timestamp >= ^start_date and
          a.timestamp <= ^end_date,
      group_by: fragment("EXTRACT(hour FROM ?)", a.timestamp),
      select: {fragment("EXTRACT(hour FROM ?)", a.timestamp), count(a.id)},
      order_by: [desc: count(a.id)],
      limit: 3
    )
    |> Repo.all()
    |> Enum.map(fn {hour, count} -> %{hour: trunc(hour), message_count: count} end)
  end

  defp calculate_room_engagement(room_id, start_date, end_date) do
    # Simple engagement score based on messages per active user
    message_count = get_room_message_count(room_id, start_date, end_date)
    active_users = get_room_active_users(room_id, start_date, end_date)

    if active_users > 0 do
      Float.round(message_count / active_users, 2)
    else
      0.0
    end
  end

  defp calculate_room_growth_trend(room_id) do
    # Compare current week vs previous week
    current_week_end = DateTime.utc_now()
    current_week_start = DateTime.add(current_week_end, -7, :day)
    previous_week_start = DateTime.add(current_week_start, -7, :day)

    current_messages = get_room_message_count(room_id, current_week_start, current_week_end)
    previous_messages = get_room_message_count(room_id, previous_week_start, current_week_start)

    if previous_messages > 0 do
      growth_rate = (current_messages - previous_messages) / previous_messages * 100
      Float.round(growth_rate, 2)
    else
      if current_messages > 0, do: 100.0, else: 0.0
    end
  end
end
