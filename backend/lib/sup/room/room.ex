defmodule Sup.Room.Room do
  @moduledoc """
  Room schema for chat rooms and group chats.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "rooms" do
    field(:name, :string)
    field(:description, :string)
    field(:type, Ecto.Enum, values: [:group, :direct_message, :channel])
    field(:is_private, :boolean, default: false)
    field(:created_by, :binary_id)
    field(:avatar_url, :string)
    field(:settings, :map, default: %{})

    # Associations
    has_many(:room_members, Sup.Room.RoomMember, foreign_key: :room_id)

    timestamps()
  end

  def changeset(room, attrs) do
    room
    |> cast(attrs, [:name, :description, :type, :is_private, :created_by, :avatar_url, :settings])
    |> validate_required([:type, :created_by])
    |> validate_length(:name, min: 1, max: 100)
    |> validate_length(:description, max: 500)
    |> validate_inclusion(:type, [:group, :direct_message, :channel])
  end

  def public_fields(room) do
    %{
      id: room.id,
      name: room.name,
      description: room.description,
      type: room.type,
      is_private: room.is_private,
      avatar_url: room.avatar_url,
      created_by: room.created_by,
      created_at: room.inserted_at,
      updated_at: room.updated_at
    }
  end
end
