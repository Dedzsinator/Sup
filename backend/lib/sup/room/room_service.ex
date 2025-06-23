defmodule Sup.Room.RoomService do
  @moduledoc """
  Room service for managing chat rooms, group chats, and member permissions.
  """

  alias Sup.Room.{Room, RoomMember}
  alias Sup.Auth.User
  alias Sup.Repo
  import Ecto.Query

  def create_room(creator_id, %{"name" => name, "type" => type} = attrs) do
    room_attrs = %{
      name: name,
      type: String.to_existing_atom(type),
      created_by: creator_id,
      description: Map.get(attrs, "description"),
      is_private: Map.get(attrs, "is_private", false)
    }

    Repo.transaction(fn ->
      # Create room
      case Room.changeset(%Room{}, room_attrs) |> Repo.insert() do
        {:ok, room} ->
          # Add creator as admin
          member_attrs = %{
            room_id: room.id,
            user_id: creator_id,
            role: :admin,
            joined_at: DateTime.utc_now()
          }

          case RoomMember.changeset(%RoomMember{}, member_attrs) |> Repo.insert() do
            {:ok, _member} ->
              broadcast_room_created(room)
              room

            {:error, changeset} ->
              Repo.rollback(changeset)
          end

        {:error, changeset} ->
          Repo.rollback(changeset)
      end
    end)
  end

  def create_direct_message(user1_id, user2_id) do
    # Check if DM already exists
    existing_dm =
      from(r in Room,
        join: rm1 in RoomMember,
        on: rm1.room_id == r.id and rm1.user_id == ^user1_id,
        join: rm2 in RoomMember,
        on: rm2.room_id == r.id and rm2.user_id == ^user2_id,
        where: r.type == :direct_message,
        select: r
      )
      |> Repo.one()

    case existing_dm do
      nil ->
        # Create new DM
        room_attrs = %{
          name: "DM",
          type: :direct_message,
          created_by: user1_id,
          is_private: true
        }

        Repo.transaction(fn ->
          case Room.changeset(%Room{}, room_attrs) |> Repo.insert() do
            {:ok, room} ->
              # Add both users as members
              members = [
                %{
                  room_id: room.id,
                  user_id: user1_id,
                  role: :member,
                  joined_at: DateTime.utc_now()
                },
                %{
                  room_id: room.id,
                  user_id: user2_id,
                  role: :member,
                  joined_at: DateTime.utc_now()
                }
              ]

              case Repo.insert_all(RoomMember, members) do
                {2, _} ->
                  room

                _ ->
                  Repo.rollback("Failed to add members")
              end

            {:error, changeset} ->
              Repo.rollback(changeset)
          end
        end)

      room ->
        {:ok, room}
    end
  end

  def join_room(user_id, room_id) do
    case get_room(room_id) do
      nil ->
        {:error, "room_not_found"}

      room ->
        if can_join_room?(user_id, room_id) do
          member_attrs = %{
            room_id: room_id,
            user_id: user_id,
            role: :member,
            joined_at: DateTime.utc_now()
          }

          case RoomMember.changeset(%RoomMember{}, member_attrs) |> Repo.insert() do
            {:ok, member} ->
              broadcast_member_joined(room, user_id)
              {:ok, member}

            {:error, changeset} ->
              {:error, changeset}
          end
        else
          {:error, "unauthorized"}
        end
    end
  end

  def leave_room(user_id, room_id) do
    case Repo.get_by(RoomMember, user_id: user_id, room_id: room_id) do
      nil ->
        {:error, "not_a_member"}

      member ->
        case Repo.delete(member) do
          {:ok, deleted_member} ->
            broadcast_member_left(room_id, user_id)
            {:ok, deleted_member}

          {:error, changeset} ->
            {:error, changeset}
        end
    end
  end

  def get_room(room_id) do
    Repo.get(Room, room_id)
  end

  def get_user_rooms(user_id) do
    from(r in Room,
      join: rm in RoomMember,
      on: rm.room_id == r.id,
      where: rm.user_id == ^user_id,
      order_by: [desc: r.updated_at],
      select: r
    )
    |> Repo.all()
  end

  def get_room_members(room_id) do
    from(u in User,
      join: rm in RoomMember,
      on: rm.user_id == u.id,
      where: rm.room_id == ^room_id,
      select: u
    )
    |> Repo.all()
  end

  def can_join_room?(user_id, room_id) do
    case get_room(room_id) do
      nil ->
        false

      room ->
        # Check if user is already a member
        existing_member = Repo.get_by(RoomMember, user_id: user_id, room_id: room_id)

        cond do
          existing_member != nil ->
            # Already a member
            true

          room.type == :direct_message ->
            # Cannot join DMs
            false

          room.is_private ->
            # Need invitation for private rooms
            false

          true ->
            # Can join public rooms
            true
        end
    end
  end

  def can_send_message?(user_id, room_id) do
    case Repo.get_by(RoomMember, user_id: user_id, room_id: room_id) do
      nil -> false
      _member -> true
    end
  end

  def can_access_room?(user_id, room_id) do
    case Repo.get_by(RoomMember, user_id: user_id, room_id: room_id) do
      nil -> false
      _member -> true
    end
  end

  def is_room_admin?(user_id, room_id) do
    case Repo.get_by(RoomMember, user_id: user_id, room_id: room_id) do
      %RoomMember{role: :admin} -> true
      %RoomMember{role: :owner} -> true
      _ -> false
    end
  end

  def update_room(room_id, user_id, attrs) do
    with room when not is_nil(room) <- get_room(room_id),
         true <- is_room_admin?(user_id, room_id) do
      case Room.changeset(room, attrs) |> Repo.update() do
        {:ok, updated_room} ->
          broadcast_room_updated(updated_room)
          {:ok, updated_room}

        {:error, changeset} ->
          {:error, changeset}
      end
    else
      nil -> {:error, "room_not_found"}
      false -> {:error, "unauthorized"}
    end
  end

  def delete_room(room_id, user_id) do
    with room when not is_nil(room) <- get_room(room_id),
         true <- room.created_by == user_id do
      case Repo.delete(room) do
        {:ok, deleted_room} ->
          broadcast_room_deleted(room_id)
          {:ok, deleted_room}

        {:error, changeset} ->
          {:error, changeset}
      end
    else
      nil -> {:error, "room_not_found"}
      false -> {:error, "unauthorized"}
    end
  end

  # Private functions
  defp broadcast_room_created(room) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "rooms", {:room_created, room})
  end

  defp broadcast_room_updated(room) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{room.id}", {:room_updated, room})
  end

  defp broadcast_room_deleted(room_id) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{room_id}", {:room_deleted, room_id})
  end

  defp broadcast_member_joined(room, user_id) do
    Phoenix.PubSub.broadcast(
      Sup.PubSub,
      "room:#{room.id}",
      {:member_joined, %{room_id: room.id, user_id: user_id}}
    )
  end

  defp broadcast_member_left(room_id, user_id) do
    Phoenix.PubSub.broadcast(
      Sup.PubSub,
      "room:#{room_id}",
      {:member_left, %{room_id: room_id, user_id: user_id}}
    )
  end
end
