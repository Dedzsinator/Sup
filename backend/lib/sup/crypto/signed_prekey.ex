defmodule Sup.Crypto.SignedPreKey do
  @moduledoc """
  Signed prekeys for Signal Protocol with signature verification.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "signed_prekeys" do
    field(:user_id, :binary_id)
    field(:key_id, :integer)
    field(:public_key, :binary)
    field(:private_key, :binary)
    field(:signature, :binary)
    field(:expires_at, :utc_datetime)

    timestamps()
  end

  def changeset(signed_prekey, attrs) do
    signed_prekey
    |> cast(attrs, [:user_id, :key_id, :public_key, :private_key, :signature, :expires_at])
    |> validate_required([:user_id, :key_id, :public_key, :private_key, :signature])
    |> unique_constraint([:user_id, :key_id])
  end
end
