defmodule Sup.Sync.MultiDeviceSyncService do
  @moduledoc """
  Service for synchronizing user data, messages, and state across multiple devices.
  Handles device registration, sync state management, and conflict resolution.
  """

  use GenServer
  require Logger

  alias Sup.Sync.DeviceSyncState
  alias Sup.Messaging.{SyncState, EnhancedMessageService}
  alias Sup.Presence.EnhancedPresenceService
  alias Sup.Repo
  import Ecto.Query

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Register a new device for a user
  """
  def register_device(user_id, device_info) do
    GenServer.call(__MODULE__, {:register_device, user_id, device_info})
  end

  @doc """
  Update device sync state
  """
  def update_sync_state(user_id, device_id, sync_data) do
    GenServer.cast(__MODULE__, {:update_sync_state, user_id, device_id, sync_data})
  end

  @doc """
  Get sync state for a device
  """
  def get_sync_state(user_id, device_id) do
    GenServer.call(__MODULE__, {:get_sync_state, user_id, device_id})
  end

  @doc """
  Synchronize messages between devices
  """
  def sync_messages(user_id, device_id, last_sync_timestamp \\ nil) do
    GenServer.call(__MODULE__, {:sync_messages, user_id, device_id, last_sync_timestamp})
  end

  @doc """
  Synchronize read receipts and message states
  """
  def sync_message_states(user_id, device_id, state_updates) do
    GenServer.cast(__MODULE__, {:sync_message_states, user_id, device_id, state_updates})
  end

  @doc """
  Synchronize presence status across devices
  """
  def sync_presence(user_id, device_id, presence_data) do
    GenServer.cast(__MODULE__, {:sync_presence, user_id, device_id, presence_data})
  end

  @doc """
  Get all devices for a user
  """
  def get_user_devices(user_id) do
    GenServer.call(__MODULE__, {:get_user_devices, user_id})
  end

  @doc """
  Remove device registration
  """
  def unregister_device(user_id, device_id) do
    GenServer.call(__MODULE__, {:unregister_device, user_id, device_id})
  end

  @doc """
  Resolve sync conflicts between devices
  """
  def resolve_conflicts(user_id, conflicts) do
    GenServer.call(__MODULE__, {:resolve_conflicts, user_id, conflicts})
  end

  @doc """
  Sync message data to other devices
  """
  def sync_message_data(user_id, data) do
    case get_user_devices(user_id) do
      {:ok, devices} ->
        Enum.each(devices, fn device ->
          send_sync_message(device.device_id, data)
        end)

        :ok

      {:error, _reason} ->
        # Fail silently for now
        :ok
    end
  end

  @doc """
  Get device state for a user
  """
  def get_device_state(user_id) do
    case get_user_devices(user_id) do
      {:ok, devices} ->
        device_states =
          Enum.map(devices, fn device ->
            %{
              device_id: device.device_id,
              last_seen: device.last_seen_at,
              is_online: device_online?(device),
              sync_state: get_device_sync_state(device.device_id)
            }
          end)

        {:ok,
         %{
           user_id: user_id,
           devices: device_states,
           total_devices: length(devices)
         }}

      error ->
        error
    end
  end

  @doc """
  Sync device state with merge strategy
  """
  def sync_device_state(user_id, device_info, strategy \\ :merge) do
    device_id = device_info["device_id"] || Map.get(device_info, :device_id)

    case get_or_create_device(user_id, device_id, device_info) do
      {:ok, device} ->
        sync_data = %{
          device_state: device_info,
          strategy: strategy,
          timestamp: DateTime.utc_now()
        }

        update_sync_state_impl(user_id, device_id, sync_data)

        {:ok, device}

      error ->
        error
    end
  end

  # GenServer callbacks

  @impl true
  def init(_opts) do
    # Schedule periodic cleanup of old sync states
    schedule_cleanup()
    {:ok, %{active_syncs: %{}}}
  end

  @impl true
  def handle_call({:register_device, user_id, device_info}, _from, state) do
    result = register_device_impl(user_id, device_info)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:get_sync_state, user_id, device_id}, _from, state) do
    result = get_sync_state_impl(user_id, device_id)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:sync_messages, user_id, device_id, last_sync_timestamp}, _from, state) do
    result = sync_messages_impl(user_id, device_id, last_sync_timestamp)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:get_user_devices, user_id}, _from, state) do
    result = get_user_devices_impl(user_id)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:unregister_device, user_id, device_id}, _from, state) do
    result = unregister_device_impl(user_id, device_id)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:resolve_conflicts, user_id, conflicts}, _from, state) do
    result = resolve_conflicts_impl(user_id, conflicts)
    {:reply, result, state}
  end

  @impl true
  def handle_cast({:update_sync_state, user_id, device_id, sync_data}, state) do
    update_sync_state_impl(user_id, device_id, sync_data)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:sync_message_states, user_id, device_id, state_updates}, state) do
    sync_message_states_impl(user_id, device_id, state_updates)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:sync_presence, user_id, device_id, presence_data}, state) do
    sync_presence_impl(user_id, device_id, presence_data)
    {:noreply, state}
  end

  @impl true
  def handle_info(:cleanup_old_states, state) do
    cleanup_old_sync_states()
    schedule_cleanup()
    {:noreply, state}
  end

  # Implementation functions

  defp register_device_impl(user_id, device_info) do
    device_id = generate_device_id(device_info)

    sync_state_attrs = %{
      user_id: user_id,
      device_id: device_id,
      device_type: device_info["type"] || "unknown",
      device_name: device_info["name"] || "Unknown Device",
      platform: device_info["platform"] || "unknown",
      app_version: device_info["app_version"] || "unknown",
      last_sync_timestamp: DateTime.utc_now(),
      sync_data: %{
        "messages" => %{"last_message_id" => nil, "last_sync" => nil},
        "read_receipts" => %{"last_sync" => nil},
        "presence" => %{"last_sync" => nil},
        "settings" => %{"last_sync" => nil}
      },
      is_active: true
    }

    case DeviceSyncState.changeset(%DeviceSyncState{}, sync_state_attrs) |> Repo.insert() do
      {:ok, sync_state} ->
        Logger.info("Registered device #{device_id} for user #{user_id}")

        # Broadcast device registration to other devices
        broadcast_device_event(user_id, device_id, :device_registered, %{
          device_name: device_info["name"],
          device_type: device_info["type"]
        })

        {:ok,
         %{
           device_id: device_id,
           sync_state: DeviceSyncState.public_fields(sync_state)
         }}

      {:error, changeset} ->
        {:error, changeset}
    end
  end

  defp get_sync_state_impl(user_id, device_id) do
    case Repo.get_by(DeviceSyncState, user_id: user_id, device_id: device_id) do
      nil ->
        {:error, "device_not_found"}

      sync_state ->
        {:ok, DeviceSyncState.public_fields(sync_state)}
    end
  end

  defp sync_messages_impl(user_id, device_id, last_sync_timestamp) do
    case get_device_sync_state(user_id, device_id) do
      nil ->
        {:error, "device_not_found"}

      sync_state ->
        # Determine sync timestamp
        sync_from =
          last_sync_timestamp ||
            get_nested_value(sync_state.sync_data, ["messages", "last_sync"])

        # Get user's rooms
        user_rooms = Sup.Room.RoomService.get_user_rooms(user_id)
        room_ids = Enum.map(user_rooms, & &1.id)

        # Get messages since last sync
        messages = get_messages_since(room_ids, sync_from)

        # Get message states (read receipts, reactions, etc.)
        message_states = get_message_states_since(user_id, sync_from)

        # Update sync state
        new_sync_timestamp = DateTime.utc_now()

        update_device_sync_data(sync_state, "messages", %{
          "last_sync" => new_sync_timestamp,
          "last_message_id" => get_latest_message_id(messages)
        })

        sync_result = %{
          messages: messages,
          message_states: message_states,
          sync_timestamp: new_sync_timestamp,
          # Pagination hint
          has_more: length(messages) >= 100
        }

        {:ok, sync_result}
    end
  end

  defp get_user_devices_impl(user_id) do
    devices =
      from(ds in DeviceSyncState,
        where: ds.user_id == ^user_id and ds.is_active == true,
        order_by: [desc: ds.last_sync_timestamp]
      )
      |> Repo.all()
      |> Enum.map(&DeviceSyncState.public_fields/1)

    {:ok, devices}
  end

  defp unregister_device_impl(user_id, device_id) do
    case Repo.get_by(DeviceSyncState, user_id: user_id, device_id: device_id) do
      nil ->
        {:error, "device_not_found"}

      sync_state ->
        case Repo.update(DeviceSyncState.changeset(sync_state, %{is_active: false})) do
          {:ok, _} ->
            Logger.info("Unregistered device #{device_id} for user #{user_id}")

            # Broadcast device removal
            broadcast_device_event(user_id, device_id, :device_unregistered, %{})

            {:ok, "device_unregistered"}

          {:error, changeset} ->
            {:error, changeset}
        end
    end
  end

  defp resolve_conflicts_impl(user_id, conflicts) do
    resolved_conflicts =
      Enum.map(conflicts, fn conflict ->
        resolve_single_conflict(user_id, conflict)
      end)

    # Broadcast resolution to all devices
    broadcast_conflict_resolution(user_id, resolved_conflicts)

    {:ok, resolved_conflicts}
  end

  defp update_sync_state_impl(user_id, device_id, sync_data) do
    case get_device_sync_state(user_id, device_id) do
      nil ->
        Logger.warning("Attempted to update sync state for unknown device #{device_id}")

      sync_state ->
        merged_sync_data = deep_merge(sync_state.sync_data || %{}, sync_data)

        changeset =
          DeviceSyncState.changeset(sync_state, %{
            sync_data: merged_sync_data,
            last_sync_timestamp: DateTime.utc_now()
          })

        case Repo.update(changeset) do
          {:ok, _updated_state} ->
            # Broadcast sync state update to other devices
            broadcast_sync_state_update(user_id, device_id, sync_data)

          {:error, changeset} ->
            Logger.error("Failed to update sync state: #{inspect(changeset)}")
        end
    end
  end

  defp sync_message_states_impl(user_id, device_id, state_updates) do
    # Process read receipts
    if state_updates["read_receipts"] do
      process_read_receipt_updates(user_id, state_updates["read_receipts"])
    end

    # Process message reactions
    if state_updates["reactions"] do
      process_reaction_updates(user_id, state_updates["reactions"])
    end

    # Process message deletions
    if state_updates["deletions"] do
      process_deletion_updates(user_id, state_updates["deletions"])
    end

    # Update device sync state
    update_device_sync_data_by_id(user_id, device_id, "message_states", %{
      "last_sync" => DateTime.utc_now()
    })

    # Broadcast to other devices
    broadcast_message_state_sync(user_id, device_id, state_updates)
  end

  defp sync_presence_impl(user_id, device_id, presence_data) do
    # Update user presence
    if presence_data["status"] do
      EnhancedPresenceService.set_user_presence(
        user_id,
        String.to_existing_atom(presence_data["status"]),
        presence_data["custom_message"]
      )
    end

    # Update activity status
    if presence_data["activity"] do
      process_activity_updates(user_id, presence_data["activity"])
    end

    # Update device sync state
    update_device_sync_data_by_id(user_id, device_id, "presence", %{
      "last_sync" => DateTime.utc_now()
    })

    # Broadcast to other devices
    broadcast_presence_sync(user_id, device_id, presence_data)
  end

  # Helper functions

  defp get_device_sync_state(user_id, device_id) do
    Repo.get_by(DeviceSyncState, user_id: user_id, device_id: device_id, is_active: true)
  end

  defp get_messages_since(_room_ids, _since_timestamp) do
    # This would query ScyllaDB or your message store
    # For now, return empty list
    []
  end

  defp get_message_states_since(_user_id, _since_timestamp) do
    # This would get read receipts, reactions, etc. since timestamp
    %{
      "read_receipts" => [],
      "reactions" => [],
      "deletions" => []
    }
  end

  defp get_latest_message_id(messages) do
    case List.last(messages) do
      %{id: id} -> id
      _ -> nil
    end
  end

  defp update_device_sync_data(sync_state, category, data) do
    current_data = sync_state.sync_data || %{}
    updated_data = put_in(current_data, [category], data)

    changeset =
      DeviceSyncState.changeset(sync_state, %{
        sync_data: updated_data,
        last_sync_timestamp: DateTime.utc_now()
      })

    Repo.update(changeset)
  end

  defp update_device_sync_data_by_id(user_id, device_id, category, data) do
    case get_device_sync_state(user_id, device_id) do
      nil -> :error
      sync_state -> update_device_sync_data(sync_state, category, data)
    end
  end

  defp resolve_single_conflict(_user_id, conflict) do
    # Implement conflict resolution logic based on conflict type
    case conflict["type"] do
      "message_edit" ->
        resolve_message_edit_conflict(conflict)

      "read_receipt" ->
        resolve_read_receipt_conflict(conflict)

      "presence_status" ->
        resolve_presence_conflict(conflict)

      _ ->
        # Default resolution: use latest timestamp
        resolve_by_timestamp(conflict)
    end
  end

  defp resolve_message_edit_conflict(conflict) do
    # Use the edit with the latest timestamp
    latest_edit =
      Enum.max_by(conflict["versions"], fn version ->
        version["timestamp"]
      end)

    %{
      conflict_id: conflict["id"],
      resolution: "latest_timestamp",
      chosen_version: latest_edit,
      resolved_at: DateTime.utc_now()
    }
  end

  defp resolve_read_receipt_conflict(conflict) do
    # Read receipts: use the earliest read timestamp
    earliest_read =
      Enum.min_by(conflict["versions"], fn version ->
        version["read_at"]
      end)

    %{
      conflict_id: conflict["id"],
      resolution: "earliest_read",
      chosen_version: earliest_read,
      resolved_at: DateTime.utc_now()
    }
  end

  defp resolve_presence_conflict(conflict) do
    # Presence: use latest status change
    latest_presence =
      Enum.max_by(conflict["versions"], fn version ->
        version["changed_at"]
      end)

    %{
      conflict_id: conflict["id"],
      resolution: "latest_change",
      chosen_version: latest_presence,
      resolved_at: DateTime.utc_now()
    }
  end

  defp resolve_by_timestamp(conflict) do
    latest_version =
      Enum.max_by(conflict["versions"], fn version ->
        version["timestamp"]
      end)

    %{
      conflict_id: conflict["id"],
      resolution: "latest_timestamp",
      chosen_version: latest_version,
      resolved_at: DateTime.utc_now()
    }
  end

  defp process_read_receipt_updates(user_id, read_receipts) do
    Enum.each(read_receipts, fn receipt ->
      EnhancedMessageService.mark_message_read(receipt["message_id"], user_id)
    end)
  end

  defp process_reaction_updates(user_id, reactions) do
    Enum.each(reactions, fn reaction ->
      case reaction["action"] do
        "add" ->
          EnhancedMessageService.add_reaction(user_id, reaction["message_id"], reaction["emoji"])

        "remove" ->
          EnhancedMessageService.remove_reaction(
            user_id,
            reaction["message_id"],
            reaction["emoji"]
          )
      end
    end)
  end

  defp process_deletion_updates(user_id, deletions) do
    Enum.each(deletions, fn deletion ->
      EnhancedMessageService.delete_message(user_id, deletion["message_id"])
    end)
  end

  defp process_activity_updates(user_id, activity_updates) do
    Enum.each(activity_updates, fn activity ->
      case activity["action"] do
        "start" ->
          EnhancedPresenceService.set_user_activity(
            user_id,
            activity["room_id"],
            String.to_existing_atom(activity["type"]),
            activity["metadata"] || %{}
          )

        "stop" ->
          EnhancedPresenceService.clear_user_activity(
            user_id,
            activity["room_id"],
            String.to_existing_atom(activity["type"])
          )
      end
    end)
  end

  defp broadcast_device_event(user_id, device_id, event_type, data) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{user_id}:devices", {
      event_type,
      %{device_id: device_id, data: data, timestamp: DateTime.utc_now()}
    })
  end

  defp broadcast_sync_state_update(user_id, device_id, sync_data) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{user_id}:sync", {
      :sync_state_updated,
      %{device_id: device_id, sync_data: sync_data, timestamp: DateTime.utc_now()}
    })
  end

  defp broadcast_message_state_sync(user_id, device_id, state_updates) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{user_id}:sync", {
      :message_states_synced,
      %{device_id: device_id, updates: state_updates, timestamp: DateTime.utc_now()}
    })
  end

  defp broadcast_presence_sync(user_id, device_id, presence_data) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{user_id}:sync", {
      :presence_synced,
      %{device_id: device_id, presence: presence_data, timestamp: DateTime.utc_now()}
    })
  end

  defp broadcast_conflict_resolution(user_id, resolved_conflicts) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{user_id}:sync", {
      :conflicts_resolved,
      %{conflicts: resolved_conflicts, timestamp: DateTime.utc_now()}
    })
  end

  defp cleanup_old_sync_states do
    # Remove sync states for devices that haven't synced in 30 days
    cutoff_date = DateTime.utc_now() |> DateTime.add(-30 * 24 * 3600, :second)

    {deleted_count, _} =
      from(ds in DeviceSyncState,
        where: ds.last_sync_timestamp < ^cutoff_date
      )
      |> Repo.delete_all()

    if deleted_count > 0 do
      Logger.info("Cleaned up #{deleted_count} old device sync states")
    end
  end

  defp schedule_cleanup do
    # Schedule cleanup every 6 hours
    Process.send_after(self(), :cleanup_old_states, :timer.hours(6))
  end

  defp generate_device_id(device_info) do
    # Generate a unique device ID based on device characteristics
    device_string = "#{device_info["type"]}_#{device_info["platform"]}_#{device_info["name"]}"
    hash = :crypto.hash(:sha256, device_string) |> Base.encode64() |> binary_part(0, 16)
    "device_#{hash}"
  end

  defp get_nested_value(map, keys) when is_map(map) do
    Enum.reduce(keys, map, fn key, acc ->
      if is_map(acc), do: Map.get(acc, key), else: nil
    end)
  end

  defp deep_merge(map1, map2) when is_map(map1) and is_map(map2) do
    Map.merge(map1, map2, fn _key, v1, v2 ->
      if is_map(v1) and is_map(v2) do
        deep_merge(v1, v2)
      else
        v2
      end
    end)
  end

  defp deep_merge(_map1, map2), do: map2

  defp send_sync_message(device_id, data) do
    channel_name = "device:#{device_id}"

    Phoenix.PubSub.broadcast(Sup.PubSub, channel_name, {
      :sync_message,
      data
    })
  end

  defp device_online?(device) do
    case device.last_seen_at do
      nil ->
        false

      last_seen ->
        diff = DateTime.diff(DateTime.utc_now(), last_seen, :minute)
        # Consider online if seen within last 5 minutes
        diff < 5
    end
  end

  defp get_device_sync_state(device_id) do
    # Get the last sync timestamp for this device
    case Repo.get_by(SyncState, device_id: device_id) do
      nil ->
        %{last_sync: nil, pending_items: 0}

      sync_state ->
        %{
          last_sync: sync_state.last_sync_at,
          pending_items: sync_state.pending_count || 0
        }
    end
  end

  @doc """
  Remove a device from a user's device list
  """
  def remove_device(user_id, device_id) do
    case Repo.get_by(DeviceSyncState, user_id: user_id, device_id: device_id) do
      nil ->
        {:error, "device_not_found"}

      device ->
        case Repo.delete(device) do
          {:ok, _deleted_device} ->
            # Clean up any sync state for this device
            from(s in SyncState, where: s.device_id == ^device_id)
            |> Repo.delete_all()

            # Broadcast device removal to user's other devices
            broadcast_to_user_devices(user_id, %{
              type: "device_removed",
              device_id: device_id,
              timestamp: DateTime.utc_now()
            })

            Logger.info("Removed device #{device_id} for user #{user_id}")
            {:ok, "device_removed"}

          {:error, changeset} ->
            {:error, changeset}
        end
    end
  end

  defp broadcast_to_user_devices(user_id, message) do
    # Get all active devices for the user
    case get_user_devices(user_id) do
      {:ok, devices} ->
        Enum.each(devices, fn device ->
          send_sync_message(device.device_id, message)
        end)

      {:error, _reason} ->
        # Fail silently
        :ok
    end
  end

  defp get_or_create_device(user_id, device_id, device_info) do
    case Repo.get_by(DeviceSyncState, user_id: user_id, device_id: device_id) do
      nil ->
        # Create new device
        device_attrs = %{
          user_id: user_id,
          device_id: device_id,
          device_name: device_info["device_name"] || "Unknown Device",
          device_type: device_info["device_type"] || "unknown",
          is_active: true,
          last_seen_at: DateTime.utc_now(),
          registered_at: DateTime.utc_now()
        }

        DeviceSyncState.changeset(%DeviceSyncState{}, device_attrs)
        |> Repo.insert()

      device ->
        # Update existing device
        device
        |> DeviceSyncState.changeset(%{
          last_seen_at: DateTime.utc_now(),
          is_active: true
        })
        |> Repo.update()
    end
  end
end
