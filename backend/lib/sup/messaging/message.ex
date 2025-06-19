defmodule Sup.Messaging.Message do
  @moduledoc """
  Message schema - stored primarily in ScyllaDB for performance.
  This is mainly for type definitions and validation.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: false}

  schema "messages" do
    field(:sender_id, :binary_id)
    field(:room_id, :binary_id)
    field(:content, :string)
    field(:type, Ecto.Enum, values: [:text, :image, :file, :audio, :video])
    field(:timestamp, :utc_datetime)
    field(:edited_at, :utc_datetime)
    field(:reply_to_id, :binary_id)

    # Encryption support (future)
    field(:encrypted_content, :binary)
    field(:encryption_key_id, :string)
  end

  def changeset(message, attrs) do
    message
    |> cast(attrs, [:sender_id, :room_id, :content, :type, :timestamp, :reply_to_id])
    |> validate_required([:sender_id, :room_id, :content, :type, :timestamp])
    |> validate_length(:content, max: 4096)
  end
end
