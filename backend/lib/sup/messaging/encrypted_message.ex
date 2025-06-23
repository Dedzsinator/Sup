defmodule Sup.Messaging.EncryptedMessage do
  @moduledoc """
  Encrypted message structure for Signal Protocol.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "encrypted_messages" do
    field(:sender_id, :binary_id)
    field(:recipient_id, :binary_id)
    field(:session_id, :binary_id)
    field(:message_number, :integer)
    field(:ciphertext, :binary)
    field(:auth_tag, :binary)
    field(:ephemeral_key, :binary)
    field(:decrypted, :boolean, default: false)

    timestamps()
  end

  def changeset(encrypted_message, attrs) do
    encrypted_message
    |> cast(attrs, [
      :sender_id, :recipient_id, :session_id, :message_number,
      :ciphertext, :auth_tag, :ephemeral_key, :decrypted
    ])
    |> validate_required([
      :sender_id, :recipient_id, :session_id, :message_number,
      :ciphertext, :auth_tag
    ])
  end
end
