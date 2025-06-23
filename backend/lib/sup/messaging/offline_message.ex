defmodule Sup.Messaging.OfflineMessage do
  @moduledoc """
  Schema for offline messages that need to be delivered when user comes online.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "offline_messages" do
    field(:user_id, :binary_id)
    field(:message_type, :string)
    field(:message_id, :binary_id)
    field(:room_id, :binary_id)
    field(:sender_id, :binary_id)
    field(:content, :string)
    field(:metadata, :map, default: %{})
    field(:priority, :integer, default: 1)
    field(:expires_at, :utc_datetime)
    field(:delivered_at, :utc_datetime)

    timestamps()
  end

  def changeset(offline_message, attrs) do
    offline_message
    |> cast(attrs, [
      :user_id,
      :message_type,
      :message_id,
      :room_id,
      :sender_id,
      :content,
      :metadata,
      :priority,
      :expires_at,
      :delivered_at
    ])
    |> validate_required([:user_id, :message_type, :content])
    |> validate_inclusion(:message_type, ["message", "reaction", "mention", "call"])
    |> validate_number(:priority, greater_than: 0, less_than_or_equal_to: 10)
  end

  def public_fields(offline_message) do
    %{
      id: offline_message.id,
      user_id: offline_message.user_id,
      message_type: offline_message.message_type,
      message_id: offline_message.message_id,
      room_id: offline_message.room_id,
      sender_id: offline_message.sender_id,
      content: offline_message.content,
      metadata: offline_message.metadata,
      priority: offline_message.priority,
      expires_at: offline_message.expires_at,
      created_at: offline_message.inserted_at
    }
  end
end
