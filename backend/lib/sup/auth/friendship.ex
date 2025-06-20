defmodule Sup.Auth.Friendship do
  @moduledoc """
  Friendship schema for managing user connections.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "friendships" do
    field(:requester_id, :binary_id)
    field(:addressee_id, :binary_id)
    field(:status, Ecto.Enum, values: [:pending, :accepted, :blocked], default: :pending)
    field(:created_at, :utc_datetime)
    field(:updated_at, :utc_datetime)
  end

  def changeset(friendship, attrs) do
    friendship
    |> cast(attrs, [:requester_id, :addressee_id, :status])
    |> validate_required([:requester_id, :addressee_id, :status])
    |> validate_inclusion(:status, [:pending, :accepted, :blocked])
    |> unique_constraint([:requester_id, :addressee_id])
    |> validate_not_self_friend()
  end

  defp validate_not_self_friend(changeset) do
    requester = get_field(changeset, :requester_id)
    addressee = get_field(changeset, :addressee_id)

    if requester && addressee && requester == addressee do
      add_error(changeset, :addressee_id, "cannot befriend yourself")
    else
      changeset
    end
  end
end
