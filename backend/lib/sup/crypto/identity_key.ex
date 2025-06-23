defmodule Sup.Crypto.IdentityKey do
  @moduledoc """
  User identity keys for Signal Protocol encryption.
  Each user has one long-term identity key pair.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "identity_keys" do
    field(:user_id, :binary_id)
    field(:public_key, :binary)
    field(:private_key, :binary)

    timestamps()
  end

  def changeset(identity_key, attrs) do
    identity_key
    |> cast(attrs, [:user_id, :public_key, :private_key])
    |> validate_required([:user_id, :public_key, :private_key])
    |> unique_constraint(:user_id)
  end
end
