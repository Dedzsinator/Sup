defmodule Sup.Messaging.MultiDeviceSyncService do
  @moduledoc """
  Service for synchronizing message state across multiple user devices.
  Handles device registration, state synchronization, and cross-device notifications.
  """

  use GenServer
  require Logger

  alias Sup.Messaging.{Message, DeviceSync, SyncState}
  alias Sup.Presence.EnhancedPresenceService
  alias Sup.Repo
  import Ecto.Query

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Register a device for a user
  """
  def register_device(user_id, device_info) do
    GenServer.call(__MODULE__, {:register_device, user_id, device_info})
  end

  @doc """
  Unregister a device
  """
  def unregister_device(user_id, device_id) do
    GenServer.call(__MODULE__, {:unregister_device, user_id, device_id})
  end

  @doc """
  Get all devices for a user
  """
  def get_user_devices(user_id) do
    GenServer.call(__MODULE__, {:get_user_devices, user_id})
  end

  @doc """
  Sync device state
  """
  def sync_device_state(user_id, device_id, state_data) do
    GenServer.cast(__MODULE__, {:sync_device_state, user_id, device_id, state_data})
  end

  @doc """
  Request sync from other devices
  """
  def request_sync(user_id, device_id, sync_types) do
    GenServer.cast(__MODULE__, {:request_sync, user_id, device_id, sync_types})
  end

  @doc """
  Handle message read across devices
  """
  def sync_message_read(user_id, message_id, room_id) do
    GenServer.cast(__MODULE__, {:sync_message_read, user_id, message_id, room_id})
  end

  @doc """
  Handle typing indicators across devices
  """
  def sync_typing_indicator(user_id, room_id, is_typing) do
    GenServer.cast(__MODULE__, {:sync_typing, user_id, room_id, is_typing})
  end

  @doc """
  Sync presence state across devices
  """
  def sync_presence_state(user_id, presence_data) do
    GenServer.cast(__MODULE__, {:sync_presence, user_id, presence_data})
  end

  # GenServer callbacks

  @impl true
  def init(_opts) do
    Logger.info("Multi-Device Sync Service started")
    {:ok, %{}}
  end

  @impl true
  def handle_call({:register_device, user_id, device_info}, _from, state) do
    result = register_device_impl(user_id, device_info)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:unregister_device, user_id, device_id}, _from, state) do
    result = unregister_device_impl(user_id, device_id)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:get_user_devices, user_id}, _from, state) do
    result = get_user_devices_impl(user_id)
    {:reply, result, state}
  end

  @impl true
  def handle_cast({:sync_device_state, user_id, device_id, state_data}, state) do
    sync_device_state_impl(user_id, device_id, state_data)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:request_sync, user_id, device_id, sync_types}, state) do
    request_sync_impl(user_id, device_id, sync_types)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:sync_message_read, user_id, message_id, room_id}, state) do
    sync_message_read_impl(user_id, message_id, room_id)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:sync_typing, user_id, room_id, is_typing}, state) do
    sync_typing_impl(user_id, room_id, is_typing)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:sync_presence, user_id, presence_data}, state) do
    sync_presence_impl(user_id, presence_data)
    {:noreply, state}
  end

  # Implementation functions

  defp register_device_impl(user_id, device_info) do
    device_params = %{
      id: device_info["device_id"] || Ecto.UUID.generate(),
      user_id: user_id,
      device_name: device_info["device_name"],
      device_type: device_info["device_type"],
      platform: device_info["platform"],
      app_version: device_info["app_version"],
      push_token: device_info["push_token"],
      last_seen: DateTime.utc_now(),
      is_active: true,
      sync_settings: device_info["sync_settings"] || default_sync_settings()
    }

    case DeviceSync.changeset(%DeviceSync{}, device_params) |> Repo.insert(
      on_conflict: {:replace_all_except, [:id, :user_id, :inserted_at]},
      conflict_target: [:id]
    ) do
      {:ok, device} ->
        Logger.info("Device registered: #{device.id} for user #{user_id}")
        
        # Broadcast device registration to other devices
        broadcast_to_user_devices(user_id, device.id, %{
          type: "device_registered",
          device: DeviceSync.public_fields(device)
        })
        
        {:ok, DeviceSync.public_fields(device)}

      {:error, changeset} ->
        {:error, changeset}
    end
  end

  defp unregister_device_impl(user_id, device_id) do
    case get_device(user_id, device_id) do
      nil ->
        {:error, "device_not_found"}

      device ->
        case Repo.update(DeviceSync.changeset(device, %{is_active: false, last_seen: DateTime.utc_now()})) do
          {:ok, updated_device} ->
            Logger.info("Device unregistered: #{device_id} for user #{user_id}")
            
            # Broadcast device unregistration to other devices
            broadcast_to_user_devices(user_id, device_id, %{
              type: "device_unregistered",
              device_id: device_id
            })
            
            {:ok, DeviceSync.public_fields(updated_device)}

          {:error, changeset} ->
            {:error, changeset}
        end
    end
  end

  defp get_user_devices_impl(user_id) do
    devices = from(ds in DeviceSync,
      where: ds.user_id == ^user_id and ds.is_active == true,
      order_by: [desc: ds.last_seen]
    ) |> Repo.all()

    {:ok, Enum.map(devices, &DeviceSync.public_fields/1)}
  end

  defp sync_device_state_impl(user_id, device_id, state_data) do
    # Update device last seen
    update_device_last_seen(user_id, device_id)

    # Store sync state
    sync_state_params = %{
      user_id: user_id,
      device_id: device_id,
      state_type: state_data["type"],
      state_data: state_data["data"],
      sync_timestamp: DateTime.utc_now()
    }

    case SyncState.changeset(%SyncState{}, sync_state_params) |> Repo.insert() do
      {:ok, sync_state} ->
        Logger.debug("Device state synced: #{device_id} for user #{user_id}")
        
        # Broadcast state change to other devices
        broadcast_to_user_devices(user_id, device_id, %{
          type: "state_synced",
          sync_state: SyncState.public_fields(sync_state)
        })

      {:error, changeset} ->
        Logger.error("Failed to sync device state: #{inspect(changeset.errors)}")
    end
  end

  defp request_sync_impl(user_id, device_id, sync_types) do
    # Update device last seen
    update_device_last_seen(user_id, device_id)

    # Broadcast sync request to other devices
    broadcast_to_user_devices(user_id, device_id, %{
      type: "sync_requested",
      requesting_device: device_id,
      sync_types: sync_types,
      timestamp: DateTime.utc_now()
    })

    Logger.debug("Sync requested by device #{device_id} for user #{user_id}: #{inspect(sync_types)}")
  end

  defp sync_message_read_impl(user_id, message_id, room_id) do
    # Update message read status for all user's devices
    broadcast_to_user_devices(user_id, nil, %{
      type: "message_read_synced",
      message_id: message_id,
      room_id: room_id,
      user_id: user_id,
      timestamp: DateTime.utc_now()
    })

    Logger.debug("Message read synced: #{message_id} in room #{room_id} for user #{user_id}")
  end

  defp sync_typing_impl(user_id, room_id, is_typing) do
    # Don't sync typing to the same device that initiated it
    broadcast_to_user_devices(user_id, nil, %{
      type: "typing_synced",
      room_id: room_id,
      user_id: user_id,
      is_typing: is_typing,
      timestamp: DateTime.utc_now()
    })

    Logger.debug("Typing synced: user #{user_id} in room #{room_id} - #{is_typing}")
  end

  defp sync_presence_impl(user_id, presence_data) do
    # Update presence across all devices
    broadcast_to_user_devices(user_id, nil, %{
      type: "presence_synced",
      user_id: user_id,
      presence_data: presence_data,
      timestamp: DateTime.utc_now()
    })

    Logger.debug("Presence synced for user #{user_id}: #{inspect(presence_data)}")
  end

  # Helper functions

  defp get_device(user_id, device_id) do
    from(ds in DeviceSync,
      where: ds.user_id == ^user_id and ds.id == ^device_id and ds.is_active == true
    ) |> Repo.one()
  end

  defp update_device_last_seen(user_id, device_id) do
    from(ds in DeviceSync,
      where: ds.user_id == ^user_id and ds.id == ^device_id
    )
    |> Repo.update_all(set: [last_seen: DateTime.utc_now()])
  end

  defp broadcast_to_user_devices(user_id, excluding_device_id, message) do
    # Get all active devices for the user
    devices = from(ds in DeviceSync,
      where: ds.user_id == ^user_id and ds.is_active == true,
      select: ds.id
    ) |> Repo.all()

    # Filter out the excluding device if specified
    target_devices = if excluding_device_id do
      Enum.reject(devices, &(&1 == excluding_device_id))
    else
      devices
    end

    # Broadcast to each device via WebSocket
    Enum.each(target_devices, fn device_id ->
      channel_name = "device:#{device_id}"
      SupWeb.Endpoint.broadcast(channel_name, "sync_message", message)
    end)
  end

  defp default_sync_settings do
    %{
      "sync_messages" => true,
      "sync_read_status" => true,
      "sync_typing" => true,
      "sync_presence" => true,
      "sync_settings" => true
    }
  end
end
