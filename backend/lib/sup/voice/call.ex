defmodule Sup.Voice.Call do
  @moduledoc """
  Call schema for managing voice and video calls.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "calls" do
    field(:caller_id, :binary_id)
    field(:room_id, :binary_id)
    field(:type, Ecto.Enum, values: [:voice, :video, :screen_share], default: :voice)

    field(:status, Ecto.Enum,
      values: [:connecting, :ringing, :active, :ended, :missed, :declined],
      default: :connecting
    )

    field(:started_at, :utc_datetime)
    field(:ended_at, :utc_datetime)
    field(:duration, :integer)
    field(:quality_metrics, :map, default: %{})
    field(:participants, {:array, :binary_id}, default: [])
    field(:recording_url, :string)
    field(:encryption_key, :string)

    timestamps()
  end

  def changeset(call, attrs) do
    call
    |> cast(attrs, [
      :caller_id,
      :room_id,
      :type,
      :status,
      :started_at,
      :ended_at,
      :duration,
      :quality_metrics,
      :participants,
      :recording_url
    ])
    |> validate_required([:caller_id, :type, :status])
    |> validate_inclusion(:type, [:voice, :video, :screen_share])
    |> validate_inclusion(:status, [:connecting, :ringing, :active, :ended, :missed, :declined])
    |> put_encryption_key()
  end

  def public_fields(call) do
    %{
      id: call.id,
      caller_id: call.caller_id,
      room_id: call.room_id,
      type: call.type,
      status: call.status,
      started_at: call.started_at,
      ended_at: call.ended_at,
      duration: call.duration,
      participants: call.participants
    }
  end

  defp put_encryption_key(changeset) do
    if get_field(changeset, :encryption_key) do
      changeset
    else
      key = :crypto.strong_rand_bytes(32) |> Base.encode64()
      put_change(changeset, :encryption_key, key)
    end
  end
end
