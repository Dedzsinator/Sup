defmodule Sup.Presence.PresenceService do
  @moduledoc """
  Presence service for tracking user online status and typing indicators.
  Uses GenServer and ETS for fast lookups with Redis backup.
  """

  use GenServer
  require Logger

  alias Sup.Redis

  @table_name :presence_table
  @typing_table :typing_table

  # Client API
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def user_online(user_id, connection_id) do
    GenServer.cast(__MODULE__, {:user_online, user_id, connection_id})
  end

  def user_offline(user_id, connection_id) do
    GenServer.cast(__MODULE__, {:user_offline, user_id, connection_id})
  end

  def user_typing(user_id, room_id, is_typing) do
    GenServer.cast(__MODULE__, {:user_typing, user_id, room_id, is_typing})
  end

  def get_online_users(room_id) do
    GenServer.call(__MODULE__, {:get_online_users, room_id})
  end

  def get_typing_users(room_id) do
    GenServer.call(__MODULE__, {:get_typing_users, room_id})
  end

  def is_user_online?(user_id) do
    case :ets.lookup(@table_name, user_id) do
      [{^user_id, _connections, _last_seen}] -> true
      [] -> false
    end
  end

  # Server callbacks
  @impl true
  def init(_opts) do
    # Create ETS tables for fast lookups
    :ets.new(@table_name, [:set, :public, :named_table])
    :ets.new(@typing_table, [:set, :public, :named_table])

    # Schedule periodic cleanup
    Process.send_after(self(), :cleanup, 30_000)

    {:ok, %{}}
  end

  @impl true
  def handle_cast({:user_online, user_id, connection_id}, state) do
    timestamp = DateTime.utc_now()

    case :ets.lookup(@table_name, user_id) do
      [{^user_id, connections, _last_seen}] ->
        # User already has connections, add this one
        new_connections = MapSet.put(connections, connection_id)
        :ets.insert(@table_name, {user_id, new_connections, timestamp})

      [] ->
        # First connection for this user
        connections = MapSet.new([connection_id])
        :ets.insert(@table_name, {user_id, connections, timestamp})

        # Broadcast user came online
        broadcast_presence_change(user_id, :online)
    end

    # Update Redis for persistence
    Redis.command(["HSET", "presence:#{user_id}", "status", "online", "last_seen", timestamp])

    {:noreply, state}
  end

  def handle_cast({:user_offline, user_id, connection_id}, state) do
    case :ets.lookup(@table_name, user_id) do
      [{^user_id, connections, _last_seen}] ->
        new_connections = MapSet.delete(connections, connection_id)

        if MapSet.size(new_connections) == 0 do
          # User has no more connections
          :ets.delete(@table_name, user_id)

          # Remove from typing table
          :ets.delete(@typing_table, user_id)

          # Broadcast user went offline
          broadcast_presence_change(user_id, :offline)

          # Update Redis
          timestamp = DateTime.utc_now()

          Redis.command([
            "HSET",
            "presence:#{user_id}",
            "status",
            "offline",
            "last_seen",
            timestamp
          ])
        else
          # User still has other connections
          :ets.insert(@table_name, {user_id, new_connections, DateTime.utc_now()})
        end

      [] ->
        # User wasn't tracked as online
        :ok
    end

    {:noreply, state}
  end

  def handle_cast({:user_typing, user_id, room_id, is_typing}, state) do
    typing_key = {user_id, room_id}

    if is_typing do
      :ets.insert(@typing_table, {typing_key, DateTime.utc_now()})
    else
      :ets.delete(@typing_table, typing_key)
    end

    # Broadcast typing status to room
    broadcast_typing_status(user_id, room_id, is_typing)

    {:noreply, state}
  end

  @impl true
  def handle_call({:get_online_users, room_id}, _from, state) do
    # Get room members and check which are online
    room_members = Sup.Room.RoomService.get_room_members(room_id)

    online_users =
      Enum.filter(room_members, fn member ->
        is_user_online?(member.id)
      end)

    {:reply, online_users, state}
  end

  def handle_call({:get_typing_users, room_id}, _from, state) do
    # Get all typing entries for this room
    typing_users =
      :ets.select(@typing_table, [
        {{{:"$1", :"$2"}, :"$3"}, [{:==, :"$2", room_id}], [:"$1"]}
      ])

    {:reply, typing_users, state}
  end

  @impl true
  def handle_info(:cleanup, state) do
    # Clean up stale typing indicators (older than 10 seconds)
    cutoff_timestamp = DateTime.add(DateTime.utc_now(), -10, :second)
    cutoff_unix = DateTime.to_unix(cutoff_timestamp, :microsecond)

    # Get all entries and filter them manually since ETS doesn't handle DateTime comparisons well
    all_entries = :ets.tab2list(@typing_table)

    stale_keys =
      all_entries
      |> Enum.filter(fn {_key, timestamp} ->
        case timestamp do
          %DateTime{} = dt -> DateTime.compare(dt, cutoff_timestamp) == :lt
          unix_time when is_integer(unix_time) -> unix_time < cutoff_unix
          # Clean up invalid entries
          _ -> true
        end
      end)
      |> Enum.map(fn {key, _timestamp} -> key end)

    Enum.each(stale_keys, fn key ->
      :ets.delete(@typing_table, key)
    end)

    # Schedule next cleanup
    Process.send_after(self(), :cleanup, 30_000)

    {:noreply, state}
  end

  # Private functions
  defp broadcast_presence_change(user_id, status) do
    # Get user's rooms to broadcast presence change
    user_rooms = Sup.Room.RoomService.get_user_rooms(user_id)

    Enum.each(user_rooms, fn room ->
      Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{room.id}", {
        :presence,
        %{user_id: user_id, status: status, timestamp: DateTime.utc_now()}
      })
    end)
  end

  defp broadcast_typing_status(user_id, room_id, is_typing) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{room_id}", {
      :typing,
      %{user_id: user_id, is_typing: is_typing, timestamp: DateTime.utc_now()}
    })
  end
end
