defmodule Sup.Messaging.DeviceSync do
  @moduledoc """
  Schema for device synchronization state.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "device_sync_state" do
    field(:user_id, :binary_id)
    field(:device_id, :string)
    field(:device_type, :string)
    field(:device_name, :string)
    field(:platform, :string)
    field(:app_version, :string)
    field(:push_token, :string)
    field(:last_sync_timestamp, :utc_datetime)
    field(:sync_state, :map, default: %{})
    field(:unread_counts, :map, default: %{})
    field(:sync_settings, :map, default: %{})
    field(:last_seen, :utc_datetime)
    field(:is_active, :boolean, default: true)

    timestamps()
  end

  def changeset(device_sync, attrs) do
    device_sync
    |> cast(attrs, [
      :user_id,
      :device_id,
      :device_type,
      :device_name,
      :platform,
      :app_version,
      :push_token,
      :last_sync_timestamp,
      :sync_state,
      :unread_counts,
      :sync_settings,
      :last_seen,
      :is_active
    ])
    |> validate_required([:user_id, :device_id, :device_type])
    |> validate_inclusion(:device_type, ["web", "mobile", "desktop"])
    |> unique_constraint([:user_id, :device_id])
  end

  def public_fields(device_sync) do
    %{
      id: device_sync.id,
      device_id: device_sync.device_id,
      device_name: device_sync.device_name,
      device_type: device_sync.device_type,
      platform: device_sync.platform,
      app_version: device_sync.app_version,
      last_sync_timestamp: device_sync.last_sync_timestamp,
      last_seen: device_sync.last_seen,
      is_active: device_sync.is_active,
      created_at: device_sync.inserted_at
    }
  end
end
