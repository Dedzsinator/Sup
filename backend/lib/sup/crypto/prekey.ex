defmodule Sup.Crypto.PreKey do
  @moduledoc """
  One-time prekeys for Signal Protocol key exchange.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "prekeys" do
    field(:user_id, :binary_id)
    field(:key_id, :integer)
    field(:public_key, :binary)
    field(:private_key, :binary)
    field(:used_at, :utc_datetime)

    timestamps()
  end

  def changeset(prekey, attrs) do
    prekey
    |> cast(attrs, [:user_id, :key_id, :public_key, :private_key, :used_at])
    |> validate_required([:user_id, :key_id, :public_key, :private_key])
    |> unique_constraint([:user_id, :key_id])
  end
end
