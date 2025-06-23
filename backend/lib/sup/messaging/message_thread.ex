defmodule Sup.Messaging.MessageThread do
  @moduledoc """
  Message thread metadata for threaded conversations.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: false}
  @foreign_key_type :binary_id

  schema "message_threads" do
    field(:room_id, :binary_id)
    field(:parent_message_id, :binary_id)
    field(:created_by, :binary_id)
    field(:message_count, :integer, default: 1)
    field(:participants, {:array, :binary_id}, default: [])
    field(:last_message_id, :binary_id)
    field(:last_activity_at, :utc_datetime)
    field(:is_pinned, :boolean, default: false)
    field(:metadata, :map, default: %{})

    timestamps()
  end

  def changeset(thread, attrs) do
    thread
    |> cast(attrs, [
      :id, :room_id, :parent_message_id, :created_by, :message_count,
      :participants, :last_message_id, :last_activity_at, :is_pinned, :metadata
    ])
    |> validate_required([:room_id, :created_by])
    |> validate_number(:message_count, greater_than: 0)
  end
end
