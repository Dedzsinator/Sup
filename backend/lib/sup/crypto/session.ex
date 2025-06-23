defmodule Sup.Crypto.Session do
  @moduledoc """
  Signal Protocol session state for Double Ratchet encryption.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "crypto_sessions" do
    field(:sender_id, :binary_id)
    field(:recipient_id, :binary_id)
    field(:root_key, :binary)
    field(:chain_key_send, :binary)
    field(:chain_key_recv, :binary)
    field(:message_number_send, :integer, default: 0)
    field(:message_number_recv, :integer, default: 0)
    field(:ephemeral_public, :binary)
    field(:ephemeral_private, :binary)
    field(:previous_counter, :integer, default: 0)

    timestamps()
  end

  def changeset(session, attrs) do
    session
    |> cast(attrs, [
      :sender_id,
      :recipient_id,
      :root_key,
      :chain_key_send,
      :chain_key_recv,
      :message_number_send,
      :message_number_recv,
      :ephemeral_public,
      :ephemeral_private,
      :previous_counter
    ])
    |> validate_required([
      :sender_id,
      :recipient_id,
      :root_key,
      :chain_key_send,
      :chain_key_recv
    ])
    |> unique_constraint([:sender_id, :recipient_id])
  end
end
