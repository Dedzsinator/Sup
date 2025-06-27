defmodule Sup.Sync.DeviceSyncState do
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "device_sync_states" do
    field(:device_id, :string)
    field(:last_sync_timestamp, :utc_datetime)
    field(:sync_state, :map, default: %{})
    field(:device_type, :string)
    field(:device_name, :string)
    field(:is_active, :boolean, default: true)

    belongs_to(:user, Sup.Auth.User)

    timestamps()
  end

  def changeset(sync_state, attrs) do
    sync_state
    |> cast(attrs, [
      :device_id,
      :last_sync_timestamp,
      :sync_state,
      :device_type,
      :device_name,
      :is_active,
      :user_id
    ])
    |> validate_required([:device_id, :device_type, :user_id])
    |> validate_inclusion(:device_type, ["mobile", "desktop", "web", "tablet"])
    |> unique_constraint([:user_id, :device_id])
    |> foreign_key_constraint(:user_id)
  end

  def public_fields(sync_state) do
    %{
      id: sync_state.id,
      device_id: sync_state.device_id,
      device_type: sync_state.device_type,
      device_name: sync_state.device_name,
      is_active: sync_state.is_active,
      last_sync_timestamp: sync_state.last_sync_timestamp,
      inserted_at: sync_state.inserted_at,
      updated_at: sync_state.updated_at
    }
  end
end
