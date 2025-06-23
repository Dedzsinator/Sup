defmodule Sup.Messaging.SyncState do
  @moduledoc """
  Schema for tracking sync state between devices.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "sync_states" do
    field(:user_id, :binary_id)
    field(:device_id, :string)
    field(:state_type, :string)
    field(:state_data, :map)
    field(:sync_timestamp, :utc_datetime)

    timestamps()
  end

  def changeset(sync_state, attrs) do
    sync_state
    |> cast(attrs, [:user_id, :device_id, :state_type, :state_data, :sync_timestamp])
    |> validate_required([:user_id, :device_id, :state_type, :state_data, :sync_timestamp])
    |> validate_inclusion(:state_type, ["messages", "read_status", "typing", "presence", "settings"])
  end

  def public_fields(sync_state) do
    %{
      id: sync_state.id,
      user_id: sync_state.user_id,
      device_id: sync_state.device_id,
      state_type: sync_state.state_type,
      state_data: sync_state.state_data,
      sync_timestamp: sync_state.sync_timestamp,
      created_at: sync_state.inserted_at
    }
  end
end
