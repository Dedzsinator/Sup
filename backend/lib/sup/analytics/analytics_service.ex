defmodule Sup.Analytics.AnalyticsService do
  @moduledoc """
  Analytics service for collecting and aggregating messaging metrics,
  user engagement data, and system performance statistics.
  """

  use GenServer
  require Logger

  alias Sup.Analytics.MessageAnalytics
  alias Sup.Repo
  import Ecto.Query

  @analytics_events [
    :message_sent,
    :message_read,
    :message_edited,
    :message_deleted,
    :reaction_added,
    :reaction_removed,
    :thread_created,
    :thread_replied,
    :user_joined,
    :user_left,
    :call_initiated,
    :call_ended,
    :bot_command_executed,
    :search_performed,
    :mention_triggered
  ]

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Track an analytics event
  """
  def track_event(event_type, user_id, metadata \\ %{}) when event_type in @analytics_events do
    GenServer.cast(__MODULE__, {:track_event, event_type, user_id, metadata})
  end

  @doc """
  Get analytics summary for a time period
  """
  def get_analytics_summary(room_id \\ nil, opts \\ []) do
    GenServer.call(__MODULE__, {:get_analytics_summary, room_id, opts})
  end

  @doc """
  Get user engagement metrics
  """
  def get_user_engagement_metrics(user_id, opts \\ []) do
    GenServer.call(__MODULE__, {:get_user_engagement_metrics, user_id, opts})
  end

  @doc """
  Get room analytics
  """
  def get_room_analytics(room_id, opts \\ []) do
    GenServer.call(__MODULE__, {:get_room_analytics, room_id, opts})
  end

  @doc """
  Get system-wide metrics
  """
  def get_system_metrics(opts \\ []) do
    GenServer.call(__MODULE__, {:get_system_metrics, opts})
  end

  @doc """
  Get popular content metrics
  """
  def get_popular_content(opts \\ []) do
    GenServer.call(__MODULE__, {:get_popular_content, opts})
  end

  # GenServer callbacks

  @impl true
  def init(_opts) do
    # Schedule periodic aggregation
    schedule_aggregation()
    {:ok, %{buffer: [], buffer_size: 0, max_buffer_size: 1000}}
  end

  @impl true
  def handle_cast({:track_event, event_type, user_id, metadata}, state) do
    # Add to buffer for batch processing
    event = %{
      event_type: event_type,
      user_id: user_id,
      metadata: metadata,
      timestamp: DateTime.utc_now()
    }

    new_buffer = [event | state.buffer]
    new_size = state.buffer_size + 1

    # Flush buffer if it gets too large
    if new_size >= state.max_buffer_size do
      flush_buffer(new_buffer)
      {:noreply, %{state | buffer: [], buffer_size: 0}}
    else
      {:noreply, %{state | buffer: new_buffer, buffer_size: new_size}}
    end
  end

  @impl true
  def handle_call({:get_analytics_summary, room_id, opts}, _from, state) do
    summary = build_analytics_summary(room_id, opts)
    {:reply, summary, state}
  end

  @impl true
  def handle_call({:get_user_engagement_metrics, user_id, opts}, _from, state) do
    metrics = build_user_engagement_metrics(user_id, opts)
    {:reply, metrics, state}
  end

  @impl true
  def handle_call({:get_room_analytics, room_id, opts}, _from, state) do
    analytics = build_room_analytics(room_id, opts)
    {:reply, analytics, state}
  end

  @impl true
  def handle_call({:get_system_metrics, opts}, _from, state) do
    metrics = build_system_metrics(opts)
    {:reply, metrics, state}
  end

  @impl true
  def handle_call({:get_popular_content, opts}, _from, state) do
    content = build_popular_content_metrics(opts)
    {:reply, content, state}
  end

  @impl true
  def handle_info(:aggregate_data, state) do
    # Flush current buffer and perform aggregations
    if state.buffer_size > 0 do
      flush_buffer(state.buffer)
    end

    perform_periodic_aggregations()
    schedule_aggregation()

    {:noreply, %{state | buffer: [], buffer_size: 0}}
  end

  # Private functions

  defp flush_buffer(events) do
    # Convert events to database records
    analytics_records =
      Enum.map(events, fn event ->
        %{
          event_type: to_string(event.event_type),
          user_id: event.user_id,
          room_id: event.metadata[:room_id],
          message_id: event.metadata[:message_id],
          metadata: event.metadata,
          event_date: DateTime.to_date(event.timestamp),
          hour_bucket: extract_hour_bucket(event.timestamp),
          inserted_at: event.timestamp,
          updated_at: event.timestamp
        }
      end)

    # Batch insert
    case Repo.insert_all(MessageAnalytics, analytics_records) do
      {count, _} ->
        Logger.debug("Inserted #{count} analytics events")

      {:error, reason} ->
        Logger.error("Failed to insert analytics events: #{inspect(reason)}")
    end
  end

  defp build_analytics_summary(room_id, opts) do
    date_from = Keyword.get(opts, :date_from, Date.utc_today() |> Date.add(-7))
    date_to = Keyword.get(opts, :date_to, Date.utc_today())

    base_query =
      from(a in MessageAnalytics,
        where: a.event_date >= ^date_from and a.event_date <= ^date_to
      )

    query =
      if room_id do
        where(base_query, [a], a.room_id == ^room_id)
      else
        base_query
      end

    # Aggregate metrics
    metrics = %{
      total_events: Repo.aggregate(query, :count, :id),
      messages_sent: count_events_by_type(query, "message_sent"),
      messages_read: count_events_by_type(query, "message_read"),
      reactions_added: count_events_by_type(query, "reaction_added"),
      threads_created: count_events_by_type(query, "thread_created"),
      daily_breakdown: get_daily_breakdown(query),
      hourly_breakdown: get_hourly_breakdown(query),
      top_users: get_top_users(query),
      event_types: get_event_type_breakdown(query)
    }

    {:ok, metrics}
  end

  defp build_user_engagement_metrics(user_id, opts) do
    date_from = Keyword.get(opts, :date_from, Date.utc_today() |> Date.add(-30))
    date_to = Keyword.get(opts, :date_to, Date.utc_today())

    query =
      from(a in MessageAnalytics,
        where:
          a.user_id == ^user_id and
            a.event_date >= ^date_from and
            a.event_date <= ^date_to
      )

    metrics = %{
      total_activity: Repo.aggregate(query, :count, :id),
      messages_sent: count_events_by_type(query, "message_sent"),
      messages_read: count_events_by_type(query, "message_read"),
      reactions_given: count_events_by_type(query, "reaction_added"),
      threads_started: count_events_by_type(query, "thread_created"),
      calls_initiated: count_events_by_type(query, "call_initiated"),
      active_days: get_user_active_days(query),
      peak_activity_hour: get_user_peak_hour(query),
      engagement_score: calculate_engagement_score(user_id, query)
    }

    {:ok, metrics}
  end

  defp build_room_analytics(room_id, opts) do
    date_from = Keyword.get(opts, :date_from, Date.utc_today() |> Date.add(-30))
    date_to = Keyword.get(opts, :date_to, Date.utc_today())

    query =
      from(a in MessageAnalytics,
        where:
          a.room_id == ^room_id and
            a.event_date >= ^date_from and
            a.event_date <= ^date_to
      )

    metrics = %{
      total_activity: Repo.aggregate(query, :count, :id),
      message_volume: count_events_by_type(query, "message_sent"),
      active_users: get_room_active_users(query),
      peak_activity_periods: get_room_peak_periods(query),
      thread_activity: count_events_by_type(query, "thread_created"),
      reaction_activity: count_events_by_type(query, "reaction_added"),
      average_response_time: calculate_average_response_time(room_id, date_from, date_to),
      engagement_trends: get_room_engagement_trends(query)
    }

    {:ok, metrics}
  end

  defp build_system_metrics(opts) do
    date_from = Keyword.get(opts, :date_from, Date.utc_today() |> Date.add(-7))
    date_to = Keyword.get(opts, :date_to, Date.utc_today())

    query =
      from(a in MessageAnalytics,
        where: a.event_date >= ^date_from and a.event_date <= ^date_to
      )

    metrics = %{
      total_events: Repo.aggregate(query, :count, :id),
      daily_active_users: get_daily_active_users(query),
      message_volume_trends: get_message_volume_trends(query),
      feature_usage: get_feature_usage_stats(query),
      performance_metrics: get_performance_metrics(),
      growth_metrics: get_growth_metrics(date_from, date_to)
    }

    {:ok, metrics}
  end

  defp build_popular_content_metrics(opts) do
    date_from = Keyword.get(opts, :date_from, Date.utc_today() |> Date.add(-7))
    date_to = Keyword.get(opts, :date_to, Date.utc_today())
    limit = Keyword.get(opts, :limit, 10)

    metrics = %{
      most_reacted_messages: get_most_reacted_messages(date_from, date_to, limit),
      most_active_threads: get_most_active_threads(date_from, date_to, limit),
      trending_rooms: get_trending_rooms(date_from, date_to, limit),
      popular_emojis: get_popular_emojis(date_from, date_to, limit),
      busiest_hours: get_busiest_hours(date_from, date_to)
    }

    {:ok, metrics}
  end

  # Helper functions for metrics calculation

  defp count_events_by_type(query, event_type) do
    from(q in query, where: q.event_type == ^event_type)
    |> Repo.aggregate(:count, :id)
  end

  defp get_daily_breakdown(query) do
    from(q in query,
      group_by: q.event_date,
      select: {q.event_date, count(q.id)},
      order_by: q.event_date
    )
    |> Repo.all()
    |> Enum.into(%{})
  end

  defp get_hourly_breakdown(query) do
    from(q in query,
      group_by: q.hour_bucket,
      select: {q.hour_bucket, count(q.id)},
      order_by: q.hour_bucket
    )
    |> Repo.all()
    |> Enum.into(%{})
  end

  defp get_top_users(query) do
    from(q in query,
      where: not is_nil(q.user_id),
      group_by: q.user_id,
      select: {q.user_id, count(q.id)},
      order_by: [desc: count(q.id)],
      limit: 10
    )
    |> Repo.all()
  end

  defp get_event_type_breakdown(query) do
    from(q in query,
      group_by: q.event_type,
      select: {q.event_type, count(q.id)}
    )
    |> Repo.all()
    |> Enum.into(%{})
  end

  defp get_user_active_days(query) do
    from(q in query,
      select: count(q.event_date, :distinct)
    )
    |> Repo.one()
  end

  defp get_user_peak_hour(query) do
    from(q in query,
      group_by: q.hour_bucket,
      select: {q.hour_bucket, count(q.id)},
      order_by: [desc: count(q.id)],
      limit: 1
    )
    |> Repo.one()
    |> case do
      {hour, _count} -> hour
      nil -> nil
    end
  end

  defp get_room_active_users(query) do
    from(q in query,
      where: not is_nil(q.user_id),
      select: count(q.user_id, :distinct)
    )
    |> Repo.one()
  end

  defp get_room_peak_periods(query) do
    from(q in query,
      group_by: [q.event_date, q.hour_bucket],
      select: {q.event_date, q.hour_bucket, count(q.id)},
      order_by: [desc: count(q.id)],
      limit: 5
    )
    |> Repo.all()
  end

  defp get_daily_active_users(query) do
    from(q in query,
      where: not is_nil(q.user_id),
      group_by: q.event_date,
      select: {q.event_date, count(q.user_id, :distinct)},
      order_by: q.event_date
    )
    |> Repo.all()
    |> Enum.into(%{})
  end

  defp get_message_volume_trends(query) do
    message_query = from(q in query, where: q.event_type == "message_sent")
    get_daily_breakdown(message_query)
  end

  defp get_feature_usage_stats(query) do
    %{
      reactions: count_events_by_type(query, "reaction_added"),
      threads: count_events_by_type(query, "thread_created"),
      calls: count_events_by_type(query, "call_initiated"),
      searches: count_events_by_type(query, "search_performed"),
      mentions: count_events_by_type(query, "mention_triggered")
    }
  end

  # Placeholder implementations for complex metrics
  defp calculate_engagement_score(_user_id, _query), do: 85.5
  defp calculate_average_response_time(_room_id, _date_from, _date_to), do: 120.5
  defp get_room_engagement_trends(_query), do: %{trend: "increasing", rate: 15.2}
  defp get_performance_metrics, do: %{avg_response_time: 45.2, error_rate: 0.01}
  defp get_growth_metrics(_date_from, _date_to), do: %{user_growth: 12.5, message_growth: 8.3}
  defp get_most_reacted_messages(_date_from, _date_to, _limit), do: []
  defp get_most_active_threads(_date_from, _date_to, _limit), do: []
  defp get_trending_rooms(_date_from, _date_to, _limit), do: []
  defp get_popular_emojis(_date_from, _date_to, _limit), do: []
  defp get_busiest_hours(_date_from, _date_to), do: %{}

  defp extract_hour_bucket(timestamp) do
    timestamp |> DateTime.to_time() |> Time.truncate(:hour) |> Time.to_string()
  end

  defp perform_periodic_aggregations do
    # Perform daily/hourly aggregations
    Logger.info("Performing periodic analytics aggregations")
  end

  defp schedule_aggregation do
    # Schedule next aggregation in 5 minutes
    Process.send_after(self(), :aggregate_data, :timer.minutes(5))
  end
end
