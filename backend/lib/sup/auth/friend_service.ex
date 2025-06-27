defmodule Sup.Auth.FriendService do
  @moduledoc """
  Service for managing friendships and friend requests.
  """

  alias Sup.Auth.{User, Friendship}
  alias Sup.Repo
  import Ecto.Query

  def send_friend_request(requester_id, addressee_identifier) do
    # Find addressee by username, email, or friend code
    addressee = find_user_by_identifier(addressee_identifier)

    case addressee do
      nil ->
        {:error, "user_not_found"}

      user when user.id == requester_id ->
        {:error, "cannot_befriend_yourself"}

      user ->
        # Check if friendship already exists
        existing =
          from(f in Friendship,
            where:
              (f.requester_id == ^requester_id and f.addressee_id == ^user.id) or
                (f.requester_id == ^user.id and f.addressee_id == ^requester_id)
          )
          |> Repo.one()

        case existing do
          nil ->
            create_friendship(requester_id, user.id)

          %Friendship{status: :blocked} ->
            {:error, "user_blocked"}

          %Friendship{status: :accepted} ->
            {:error, "already_friends"}

          %Friendship{status: :pending} ->
            {:error, "request_pending"}
        end
    end
  end

  def accept_friend_request(addressee_id, requester_id) do
    case get_pending_request(requester_id, addressee_id) do
      nil ->
        {:error, "request_not_found"}

      friendship ->
        friendship
        |> Friendship.changeset(%{status: :accepted, updated_at: DateTime.utc_now()})
        |> Repo.update()
    end
  end

  def decline_friend_request(addressee_id, requester_id) do
    case get_pending_request(requester_id, addressee_id) do
      nil ->
        {:error, "request_not_found"}

      friendship ->
        Repo.delete(friendship)
    end
  end

  def block_user(blocker_id, blocked_id) do
    # Remove existing friendship if any
    from(f in Friendship,
      where:
        (f.requester_id == ^blocker_id and f.addressee_id == ^blocked_id) or
          (f.requester_id == ^blocked_id and f.addressee_id == ^blocker_id)
    )
    |> Repo.delete_all()

    # Create block relationship
    %Friendship{}
    |> Friendship.changeset(%{
      requester_id: blocker_id,
      addressee_id: blocked_id,
      status: :blocked,
      created_at: DateTime.utc_now(),
      updated_at: DateTime.utc_now()
    })
    |> Repo.insert()
  end

  def unblock_user(blocker_id, blocked_id) do
    from(f in Friendship,
      where:
        f.requester_id == ^blocker_id and f.addressee_id == ^blocked_id and f.status == :blocked
    )
    |> Repo.delete_all()

    {:ok, "user_unblocked"}
  end

  def remove_friend(user_id, friend_id) do
    from(f in Friendship,
      where:
        ((f.requester_id == ^user_id and f.addressee_id == ^friend_id) or
           (f.requester_id == ^friend_id and f.addressee_id == ^user_id)) and
          f.status == :accepted
    )
    |> Repo.delete_all()

    {:ok, "friend_removed"}
  end

  def get_friends(user_id) do
    # Get all accepted friendships where user is involved
    query =
      from(f in Friendship,
        join: u in User,
        on:
          (f.requester_id == u.id and f.addressee_id == ^user_id) or
            (f.addressee_id == u.id and f.requester_id == ^user_id),
        where: f.status == :accepted and u.id != ^user_id,
        select: u
      )

    query
    |> Repo.all()
    |> Enum.map(&User.friends_fields/1)
  end

  def get_friend_requests(user_id) do
    # Get pending requests sent to this user
    query =
      from(f in Friendship,
        join: u in User,
        on: f.requester_id == u.id,
        where: f.addressee_id == ^user_id and f.status == :pending,
        select: %{request: f, user: u}
      )

    query
    |> Repo.all()
    |> Enum.map(fn %{request: request, user: user} ->
      %{
        id: request.id,
        requester: User.friends_fields(user),
        created_at: request.created_at
      }
    end)
  end

  def get_sent_requests(user_id) do
    # Get pending requests sent by this user
    query =
      from(f in Friendship,
        join: u in User,
        on: f.addressee_id == u.id,
        where: f.requester_id == ^user_id and f.status == :pending,
        select: %{request: f, user: u}
      )

    query
    |> Repo.all()
    |> Enum.map(fn %{request: request, user: user} ->
      %{
        id: request.id,
        addressee: User.friends_fields(user),
        created_at: request.created_at
      }
    end)
  end

  def get_blocked_users(user_id) do
    query =
      from(f in Friendship,
        join: u in User,
        on: f.addressee_id == u.id,
        where: f.requester_id == ^user_id and f.status == :blocked,
        select: u
      )

    query
    |> Repo.all()
    |> Enum.map(&User.friends_fields/1)
  end

  def are_friends?(user1_id, user2_id) do
    query =
      from(f in Friendship,
        where:
          ((f.requester_id == ^user1_id and f.addressee_id == ^user2_id) or
             (f.requester_id == ^user2_id and f.addressee_id == ^user1_id)) and
            f.status == :accepted
      )

    Repo.exists?(query)
  end

  def is_blocked?(blocker_id, blocked_id) do
    query =
      from(f in Friendship,
        where:
          f.requester_id == ^blocker_id and f.addressee_id == ^blocked_id and f.status == :blocked
      )

    Repo.exists?(query)
  end

  @doc """
  Respond to a friend request (accept or decline)
  """
  def respond_to_friend_request(addressee_id, requester_id, response)
      when response in [:accept, :decline] do
    case response do
      :accept -> accept_friend_request(addressee_id, requester_id)
      :decline -> decline_friend_request(addressee_id, requester_id)
    end
  end

  @doc """
  Search users by username, email, or friend code
  """
  def search_users(query, requesting_user_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 20)

    # Don't return the requesting user in search results
    search_query = "%#{String.downcase(query)}%"

    users =
      from(u in User,
        where:
          u.id != ^requesting_user_id and
            (ilike(u.username, ^search_query) or
               ilike(u.email, ^search_query) or
               ilike(u.friend_code, ^search_query)),
        limit: ^limit,
        select: [:id, :username, :email, :friend_code, :avatar_url]
      )
      |> Repo.all()

    # Check friendship status for each user
    user_ids = Enum.map(users, & &1.id)

    friendships =
      from(f in Friendship,
        where:
          (f.requester_id == ^requesting_user_id and f.addressee_id in ^user_ids) or
            (f.requester_id in ^user_ids and f.addressee_id == ^requesting_user_id)
      )
      |> Repo.all()

    # Create a map of user_id -> friendship_status
    friendship_map =
      Enum.reduce(friendships, %{}, fn friendship, acc ->
        other_user_id =
          if friendship.requester_id == requesting_user_id do
            friendship.addressee_id
          else
            friendship.requester_id
          end

        Map.put(acc, other_user_id, friendship.status)
      end)

    # Attach friendship status to each user
    users_with_status =
      Enum.map(users, fn user ->
        friendship_status = Map.get(friendship_map, user.id, :none)
        Map.put(user, :friendship_status, friendship_status)
      end)

    {:ok, users_with_status}
  end

  defp find_user_by_identifier(identifier) do
    # Try to find by username first
    user = Repo.get_by(User, username: identifier)

    if user do
      user
    else
      # Try by friend code
      user = Repo.get_by(User, friend_code: identifier)

      if user do
        user
      else
        # Try by email
        Repo.get_by(User, email: identifier)
      end
    end
  end

  defp create_friendship(requester_id, addressee_id) do
    %Friendship{}
    |> Friendship.changeset(%{
      requester_id: requester_id,
      addressee_id: addressee_id,
      status: :pending,
      created_at: DateTime.utc_now(),
      updated_at: DateTime.utc_now()
    })
    |> Repo.insert()
  end

  defp get_pending_request(requester_id, addressee_id) do
    Repo.get_by(Friendship,
      requester_id: requester_id,
      addressee_id: addressee_id,
      status: :pending
    )
  end
end
