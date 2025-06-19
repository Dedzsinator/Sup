defmodule Sup.Room.RoomMember do
  @moduledoc """
  Room member schema for tracking user membership in rooms.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "room_members" do
    field(:room_id, :binary_id)
    field(:user_id, :binary_id)
    field(:role, Ecto.Enum, values: [:member, :admin, :owner], default: :member)
    field(:joined_at, :utc_datetime)
    field(:permissions, :map, default: %{})

    timestamps()
  end

  def changeset(member, attrs) do
    member
    |> cast(attrs, [:room_id, :user_id, :role, :joined_at, :permissions])
    |> validate_required([:room_id, :user_id, :role, :joined_at])
    |> validate_inclusion(:role, [:member, :admin, :owner])
    |> unique_constraint([:room_id, :user_id])
  end
end
