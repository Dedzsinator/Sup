defmodule Sup.Presence.EnhancedPresenceService do
  @moduledoc """
  Enhanced presence service with detailed status, activity tracking,
  typing indicators, and voice/video call presence.
  """

  use GenServer
  require Logger

  alias Sup.Redis
  alias Sup.Room.RoomService

  @table_name :enhanced_presence_table
  @typing_table :enhanced_typing_table
  @activity_table :user_activity_table
  @voice_presence_table :voice_presence_table

  # Status types
  # Module attributes removed as they were unused

  # Client API
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def set_user_presence(user_id, status, custom_message \\ nil) do
    GenServer.cast(__MODULE__, {:set_presence, user_id, status, custom_message})
  end

  def set_user_activity(user_id, room_id, activity_type, metadata \\ %{}) do
    GenServer.cast(__MODULE__, {:set_activity, user_id, room_id, activity_type, metadata})
  end

  def clear_user_activity(user_id, room_id, activity_type) do
    GenServer.cast(__MODULE__, {:clear_activity, user_id, room_id, activity_type})
  end

  def set_voice_presence(user_id, room_id, call_id, call_type) do
    GenServer.cast(__MODULE__, {:set_voice_presence, user_id, room_id, call_id, call_type})
  end

  def clear_voice_presence(user_id) do
    GenServer.cast(__MODULE__, {:clear_voice_presence, user_id})
  end

  def get_user_presence(user_id) do
    GenServer.call(__MODULE__, {:get_presence, user_id})
  end

  def get_room_presence(room_id) do
    GenServer.call(__MODULE__, {:get_room_presence, room_id})
  end

  def get_user_activity(user_id, room_id) do
    GenServer.call(__MODULE__, {:get_activity, user_id, room_id})
  end

  def get_room_activities(room_id) do
    GenServer.call(__MODULE__, {:get_room_activities, room_id})
  end

  def get_online_count(room_id) do
    GenServer.call(__MODULE__, {:get_online_count, room_id})
  end

  # Server callbacks
  @impl true
  def init(_opts) do
    # Create ETS tables
    :ets.new(@table_name, [:set, :public, :named_table])
    :ets.new(@typing_table, [:bag, :public, :named_table])
    :ets.new(@activity_table, [:bag, :public, :named_table])
    :ets.new(@voice_presence_table, [:set, :public, :named_table])

    # Schedule periodic cleanup
    Process.send_after(self(), :cleanup, 30_000)
    Process.send_after(self(), :sync_redis, 60_000)

    {:ok, %{}}
  end

  @impl true
  def handle_cast({:set_presence, user_id, status, custom_message}, state) do
    timestamp = DateTime.utc_now()

    presence_data = %{
      user_id: user_id,
      status: status,
      custom_message: custom_message,
      last_seen: timestamp,
      device_info: get_device_info(user_id)
    }

    # Update ETS
    :ets.insert(@table_name, {user_id, presence_data})

    # Update Redis for persistence
    Redis.command([
      "HSET",
      "presence:#{user_id}",
      "status",
      Atom.to_string(status),
      "custom_message",
      custom_message || "",
      "last_seen",
      DateTime.to_iso8601(timestamp),
      "device_count",
      get_device_count(user_id)
    ])

    # Broadcast presence change to user's rooms
    broadcast_presence_change(user_id, presence_data)

    {:noreply, state}
  end

  def handle_cast({:set_activity, user_id, room_id, activity_type, metadata}, state) do
    timestamp = DateTime.utc_now()

    activity_data = %{
      user_id: user_id,
      room_id: room_id,
      activity_type: activity_type,
      metadata: metadata,
      started_at: timestamp
    }

    # Store in ETS (using bag table for multiple activities)
    activity_key = {user_id, room_id, activity_type}
    :ets.insert(@activity_table, {activity_key, activity_data})

    # Set TTL in Redis
    Redis.command([
      "SETEX",
      "activity:#{user_id}:#{room_id}:#{activity_type}",
      # 30 seconds TTL
      30,
      Jason.encode!(activity_data)
    ])

    # Broadcast activity
    broadcast_activity_change(user_id, room_id, activity_type, "started", metadata)

    {:noreply, state}
  end

  def handle_cast({:clear_activity, user_id, room_id, activity_type}, state) do
    activity_key = {user_id, room_id, activity_type}
    :ets.delete(@activity_table, activity_key)

    # Remove from Redis
    Redis.command(["DEL", "activity:#{user_id}:#{room_id}:#{activity_type}"])

    # Broadcast activity cleared
    broadcast_activity_change(user_id, room_id, activity_type, "stopped", %{})

    {:noreply, state}
  end

  def handle_cast({:set_voice_presence, user_id, room_id, call_id, call_type}, state) do
    voice_data = %{
      user_id: user_id,
      room_id: room_id,
      call_id: call_id,
      call_type: call_type,
      joined_at: DateTime.utc_now(),
      audio_enabled: true,
      video_enabled: call_type == :video
    }

    :ets.insert(@voice_presence_table, {user_id, voice_data})

    # Broadcast voice presence
    broadcast_voice_presence_change(user_id, room_id, "joined", voice_data)

    {:noreply, state}
  end

  def handle_cast({:clear_voice_presence, user_id}, state) do
    case :ets.lookup(@voice_presence_table, user_id) do
      [{^user_id, voice_data}] ->
        :ets.delete(@voice_presence_table, user_id)
        broadcast_voice_presence_change(user_id, voice_data.room_id, "left", voice_data)

      [] ->
        :ok
    end

    {:noreply, state}
  end

  @impl true
  def handle_call({:get_presence, user_id}, _from, state) do
    presence =
      case :ets.lookup(@table_name, user_id) do
        [{^user_id, data}] -> data
        [] -> %{status: :offline, last_seen: nil}
      end

    {:reply, presence, state}
  end

  def handle_call({:get_room_presence, room_id}, _from, state) do
    room_members = RoomService.get_room_members(room_id)

    presence_data =
      Enum.map(room_members, fn member ->
        case :ets.lookup(@table_name, member.id) do
          [{_, data}] -> Map.put(data, :user_id, member.id)
          [] -> %{user_id: member.id, status: :offline, last_seen: nil}
        end
      end)

    {:reply, presence_data, state}
  end

  def handle_call({:get_activity, user_id, room_id}, _from, state) do
    activities =
      :ets.select(@activity_table, [
        {{{user_id, room_id, :"$1"}, :"$2"}, [], [:"$2"]}
      ])

    {:reply, activities, state}
  end

  def handle_call({:get_room_activities, room_id}, _from, state) do
    activities =
      :ets.select(@activity_table, [
        {{{:"$1", room_id, :"$2"}, :"$3"}, [], [:"$3"]}
      ])

    # Group by activity type
    grouped_activities =
      Enum.group_by(activities, fn activity ->
        activity.activity_type
      end)

    {:reply, grouped_activities, state}
  end

  def handle_call({:get_online_count, room_id}, _from, state) do
    room_members = RoomService.get_room_members(room_id)

    online_count =
      Enum.count(room_members, fn member ->
        case :ets.lookup(@table_name, member.id) do
          [{_, %{status: status}}] -> status in [:online, :away, :busy]
          [] -> false
        end
      end)

    {:reply, online_count, state}
  end

  @impl true
  def handle_info(:cleanup, state) do
    # Clean up stale activities (older than 30 seconds)
    cutoff_timestamp = DateTime.add(DateTime.utc_now(), -30, :second)

    stale_activities =
      :ets.select(@activity_table, [
        {{:"$1", :"$2"}, [{:<, {:map_get, :started_at, :"$2"}, cutoff_timestamp}], [:"$1"]}
      ])

    Enum.each(stale_activities, fn key ->
      :ets.delete(@activity_table, key)
    end)

    # Schedule next cleanup
    Process.send_after(self(), :cleanup, 30_000)

    {:noreply, state}
  end

  def handle_info(:sync_redis, state) do
    # Sync presence data to Redis for persistence
    all_presence = :ets.tab2list(@table_name)

    Enum.each(all_presence, fn {user_id, presence_data} ->
      Redis.command([
        "HSET",
        "presence:#{user_id}",
        "status",
        Atom.to_string(presence_data.status),
        "last_seen",
        DateTime.to_iso8601(presence_data.last_seen)
      ])
    end)

    # Schedule next sync
    Process.send_after(self(), :sync_redis, 60_000)

    {:noreply, state}
  end

  # Private functions
  defp broadcast_presence_change(user_id, presence_data) do
    # Get user's rooms to broadcast presence change
    user_rooms = RoomService.get_user_rooms(user_id)

    Enum.each(user_rooms, fn room ->
      Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{room.id}", {
        :presence_update,
        %{
          user_id: user_id,
          status: presence_data.status,
          custom_message: presence_data.custom_message,
          last_seen: presence_data.last_seen
        }
      })
    end)
  end

  defp broadcast_activity_change(user_id, room_id, activity_type, action, metadata) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{room_id}", {
      :activity_update,
      %{
        user_id: user_id,
        activity_type: activity_type,
        action: action,
        metadata: metadata,
        timestamp: DateTime.utc_now()
      }
    })
  end

  defp broadcast_voice_presence_change(user_id, room_id, action, voice_data) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{room_id}", {
      :voice_presence_update,
      %{
        user_id: user_id,
        action: action,
        call_id: voice_data.call_id,
        call_type: voice_data.call_type,
        audio_enabled: voice_data.audio_enabled,
        video_enabled: voice_data.video_enabled
      }
    })
  end

  defp get_device_info(user_id) do
    # Get device information from registry
    case Registry.lookup(Sup.ConnectionRegistry, user_id) do
      [_ | _] = connections ->
        %{device_count: length(connections)}

      [] ->
        %{device_count: 0}
    end
  end

  defp get_device_count(user_id) do
    case Registry.lookup(Sup.ConnectionRegistry, user_id) do
      connections -> length(connections)
    end
  end
end
