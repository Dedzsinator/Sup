defmodule Sup.ApiRouter do
  @moduledoc """
  Protected API routes requiring authentication.
  """

  use Plug.Router
  require Logger

  alias Sup.Auth.{Guardian, User, FriendService}
  alias Sup.Room.RoomService
  alias Sup.Messaging.MessageService
  alias Sup.Voice.CallService
  alias Sup.Autocomplete.Service, as: AutocompleteService
  alias Sup.Security.{RateLimitPlug, AuthorizationPlug, AuditLog}

  plug(:match)
  plug(RateLimitPlug, limit_type: :api)
  plug(:authenticate)
  plug(AuthorizationPlug)
  plug(:dispatch)

  # Room management
  post "/rooms" do
    with {:ok, params} <- validate_create_room_params(conn.body_params),
         {:ok, room} <- RoomService.create_room(conn.assigns.current_user.id, params) do
      send_resp(conn, 201, Jason.encode!(room))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/rooms" do
    user_id = conn.assigns.current_user.id
    rooms = RoomService.get_user_rooms(user_id)

    rooms_data = Enum.map(rooms, &Sup.Room.Room.public_fields/1)
    send_resp(conn, 200, Jason.encode!(%{rooms: rooms_data}))
  end

  get "/rooms/:room_id" do
    room_id = conn.path_params["room_id"]
    user_id = conn.assigns.current_user.id

    case RoomService.get_room(room_id) do
      nil ->
        send_resp(conn, 404, Jason.encode!(%{error: "room_not_found"}))

      room ->
        if RoomService.can_send_message?(user_id, room_id) do
          send_resp(conn, 200, Jason.encode!(Sup.Room.Room.public_fields(room)))
        else
          send_resp(conn, 403, Jason.encode!(%{error: "unauthorized"}))
        end
    end
  end

  post "/rooms/:room_id/join" do
    room_id = conn.path_params["room_id"]
    user_id = conn.assigns.current_user.id

    case RoomService.join_room(user_id, room_id) do
      {:ok, _member} ->
        send_resp(conn, 200, Jason.encode!(%{message: "joined_successfully"}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  delete "/rooms/:room_id/leave" do
    room_id = conn.path_params["room_id"]
    user_id = conn.assigns.current_user.id

    case RoomService.leave_room(user_id, room_id) do
      {:ok, _member} ->
        send_resp(conn, 200, Jason.encode!(%{message: "left_successfully"}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/rooms/:room_id/messages" do
    room_id = conn.path_params["room_id"]
    user_id = conn.assigns.current_user.id

    # Check if user can access this room
    if RoomService.can_send_message?(user_id, room_id) do
      limit = conn.query_params["limit"] || "50"
      before_timestamp = conn.query_params["before"]

      case MessageService.get_room_messages(room_id, String.to_integer(limit), before_timestamp) do
        {:ok, messages} ->
          send_resp(conn, 200, Jason.encode!(%{messages: messages}))

        {:error, reason} ->
          send_resp(conn, 500, Jason.encode!(%{error: reason}))
      end
    else
      send_resp(conn, 403, Jason.encode!(%{error: "unauthorized"}))
    end
  end

  # Direct message creation
  post "/direct_messages" do
    with {:ok, params} <- validate_dm_params(conn.body_params),
         {:ok, room} <-
           RoomService.create_direct_message(conn.assigns.current_user.id, params["user_id"]) do
      send_resp(conn, 201, Jason.encode!(Sup.Room.Room.public_fields(room)))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  # Message search
  get "/search/messages" do
    query = conn.query_params["q"]
    limit = String.to_integer(conn.query_params["limit"] || "20")
    user_id = conn.assigns.current_user.id

    if query && String.length(query) >= 3 do
      case MessageService.search_messages(user_id, query, limit) do
        {:ok, messages} ->
          send_resp(conn, 200, Jason.encode!(%{messages: messages}))

        {:error, reason} ->
          send_resp(conn, 500, Jason.encode!(%{error: reason}))
      end
    else
      send_resp(conn, 400, Jason.encode!(%{error: "query_too_short"}))
    end
  end

  # Push token registration
  post "/push/register" do
    with {:ok, params} <- validate_push_token_params(conn.body_params),
         :ok <-
           Sup.Push.PushService.register_push_token(
             conn.assigns.current_user.id,
             params["token"],
             params["platform"]
           ) do
      send_resp(conn, 200, Jason.encode!(%{message: "token_registered"}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  delete "/push/unregister" do
    case Sup.Push.PushService.unregister_push_token(conn.assigns.current_user.id) do
      :ok ->
        send_resp(conn, 200, Jason.encode!(%{message: "token_unregistered"}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  # User profile
  get "/profile" do
    user = User.public_fields(conn.assigns.current_user)
    send_resp(conn, 200, Jason.encode!(%{user: user}))
  end

  put "/profile" do
    with {:ok, params} <- validate_profile_params(conn.body_params),
         {:ok, updated_user} <- update_user_profile(conn.assigns.current_user, params) do
      user_data = User.public_fields(updated_user)
      send_resp(conn, 200, Jason.encode!(%{user: user_data}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  # Autocomplete endpoints
  post "/autocomplete/suggest" do
    with {:ok, params} <- validate_autocomplete_params(conn.body_params),
         {:ok, suggestions} <-
           AutocompleteService.get_suggestions(
             params["text"],
             user_id: conn.assigns.current_user.id,
             room_id: params["room_id"],
             limit: params["limit"] || 5
           ) do
      send_resp(conn, 200, Jason.encode!(%{suggestions: suggestions}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  post "/autocomplete/complete" do
    with {:ok, params} <- validate_completion_params(conn.body_params),
         {:ok, completion} <-
           AutocompleteService.get_completion(
             params["text"],
             user_id: conn.assigns.current_user.id,
             room_id: params["room_id"],
             max_length: params["max_length"] || 50
           ) do
      send_resp(conn, 200, Jason.encode!(%{completion: completion}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/autocomplete/health" do
    case AutocompleteService.health_check() do
      {:ok, status} ->
        send_resp(conn, 200, Jason.encode!(status))

      {:error, reason} ->
        send_resp(conn, 503, Jason.encode!(%{error: reason}))
    end
  end

  get "/autocomplete/stats" do
    case AutocompleteService.get_stats() do
      {:ok, stats} ->
        send_resp(conn, 200, Jason.encode!(stats))

      {:error, reason} ->
        send_resp(conn, 503, Jason.encode!(%{error: reason}))
    end
  end

  # Friend management endpoints
  get "/friends" do
    user_id = conn.assigns.current_user.id

    case FriendService.get_friends(user_id) do
      {:ok, friends} ->
        send_resp(conn, 200, Jason.encode!(%{friends: friends}))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  get "/friends/requests" do
    user_id = conn.assigns.current_user.id

    case FriendService.get_friend_requests(user_id) do
      {:ok, requests} ->
        send_resp(conn, 200, Jason.encode!(%{requests: requests}))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  post "/friends/request" do
    with {:ok, params} <- validate_friend_request_params(conn.body_params),
         {:ok, request} <-
           FriendService.send_friend_request(
             conn.assigns.current_user.id,
             params["target_user_id"]
           ) do
      send_resp(conn, 201, Jason.encode!(%{request: request}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  put "/friends/request/:request_id" do
    request_id = conn.path_params["request_id"]

    with {:ok, params} <- validate_friend_response_params(conn.body_params),
         {:ok, result} <-
           FriendService.respond_to_friend_request(
             conn.assigns.current_user.id,
             request_id,
             params["action"]
           ) do
      send_resp(conn, 200, Jason.encode!(result))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  delete "/friends/:friend_id" do
    friend_id = conn.path_params["friend_id"]
    user_id = conn.assigns.current_user.id

    case FriendService.remove_friend(user_id, friend_id) do
      {:ok, _} ->
        send_resp(conn, 200, Jason.encode!(%{message: "friend_removed"}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  post "/friends/block" do
    with {:ok, params} <- validate_block_user_params(conn.body_params),
         {:ok, _} <-
           FriendService.block_user(
             conn.assigns.current_user.id,
             params["target_user_id"]
           ) do
      send_resp(conn, 200, Jason.encode!(%{message: "user_blocked"}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  delete "/friends/block/:blocked_user_id" do
    blocked_user_id = conn.path_params["blocked_user_id"]
    user_id = conn.assigns.current_user.id

    case FriendService.unblock_user(user_id, blocked_user_id) do
      {:ok, _} ->
        send_resp(conn, 200, Jason.encode!(%{message: "user_unblocked"}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/friends/blocked" do
    user_id = conn.assigns.current_user.id

    case FriendService.get_blocked_users(user_id) do
      {:ok, blocked_users} ->
        send_resp(conn, 200, Jason.encode!(%{blocked_users: blocked_users}))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  get "/users/search" do
    query = conn.query_params["q"]
    limit = String.to_integer(conn.query_params["limit"] || "10")
    user_id = conn.assigns.current_user.id

    if query && String.length(query) >= 2 do
      case FriendService.search_users(query, user_id, limit) do
        {:ok, users} ->
          send_resp(conn, 200, Jason.encode!(%{users: users}))

        {:error, reason} ->
          send_resp(conn, 500, Jason.encode!(%{error: reason}))
      end
    else
      send_resp(conn, 400, Jason.encode!(%{error: "query_too_short"}))
    end
  end

  # Call management endpoints
  post "/calls" do
    with {:ok, params} <- validate_call_params(conn.body_params),
         {:ok, call} <-
           CallService.initiate_call(
             conn.assigns.current_user.id,
             params["target_user_id"],
             params["call_type"]
           ) do
      send_resp(conn, 201, Jason.encode!(%{call: call}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  put "/calls/:call_id/accept" do
    call_id = conn.path_params["call_id"]
    user_id = conn.assigns.current_user.id

    case CallService.accept_call(call_id, user_id) do
      {:ok, call} ->
        send_resp(conn, 200, Jason.encode!(%{call: call}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  put "/calls/:call_id/reject" do
    call_id = conn.path_params["call_id"]
    user_id = conn.assigns.current_user.id

    case CallService.reject_call(call_id, user_id) do
      {:ok, call} ->
        send_resp(conn, 200, Jason.encode!(%{call: call}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  put "/calls/:call_id/end" do
    call_id = conn.path_params["call_id"]
    user_id = conn.assigns.current_user.id

    case CallService.end_call(call_id, user_id) do
      {:ok, call} ->
        send_resp(conn, 200, Jason.encode!(%{call: call}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  post "/calls/:call_id/signal" do
    call_id = conn.path_params["call_id"]
    user_id = conn.assigns.current_user.id

    with {:ok, params} <- validate_signal_params(conn.body_params),
         {:ok, _} <-
           CallService.handle_webrtc_signal(
             call_id,
             user_id,
             params["signal"]
           ) do
      send_resp(conn, 200, Jason.encode!(%{message: "signal_sent"}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/calls/active" do
    user_id = conn.assigns.current_user.id

    case CallService.get_active_calls(user_id) do
      {:ok, calls} ->
        send_resp(conn, 200, Jason.encode!(%{calls: calls}))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  # User settings endpoints
  put "/settings" do
    with {:ok, params} <- validate_settings_params(conn.body_params),
         {:ok, user} <- update_user_settings(conn.assigns.current_user, params) do
      send_resp(conn, 200, Jason.encode!(%{user: User.public_fields(user)}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/settings" do
    user = conn.assigns.current_user

    settings = %{
      notification_settings: user.notification_settings,
      privacy_settings: user.privacy_settings,
      call_settings: user.call_settings,
      theme_preference: user.theme_preference,
      accent_color: user.accent_color
    }

    send_resp(conn, 200, Jason.encode!(%{settings: settings}))
  end

  # File upload endpoints
  post "/upload/avatar" do
    # This would handle file upload for avatars
    # Implementation depends on your file storage solution (S3, local, etc.)
    send_resp(conn, 501, Jason.encode!(%{error: "not_implemented"}))
  end

  post "/upload/banner" do
    # This would handle file upload for profile banners
    # Implementation depends on your file storage solution (S3, local, etc.)
    send_resp(conn, 501, Jason.encode!(%{error: "not_implemented"}))
  end

  # Catch-all
  match _ do
    send_resp(conn, 404, Jason.encode!(%{error: "not_found"}))
  end

  # Authentication plug
  defp authenticate(conn, _opts) do
    case get_req_header(conn, "authorization") do
      ["Bearer " <> token] ->
        case Guardian.decode_and_verify(token) do
          {:ok, %{"sub" => user_id}} ->
            case Sup.Repo.get(User, user_id) do
              nil ->
                conn
                |> send_resp(401, Jason.encode!(%{error: "user_not_found"}))
                |> halt()

              user ->
                assign(conn, :current_user, user)
            end

          _ ->
            conn
            |> send_resp(401, Jason.encode!(%{error: "invalid_token"}))
            |> halt()
        end

      _ ->
        conn
        |> send_resp(401, Jason.encode!(%{error: "missing_token"}))
        |> halt()
    end
  end

  # Validation helpers
  defp validate_create_room_params(%{"name" => name, "type" => type})
       when is_binary(name) and is_binary(type) do
    {:ok, %{"name" => name, "type" => type}}
  end

  defp validate_create_room_params(_), do: {:error, "invalid_params"}

  defp validate_dm_params(%{"user_id" => user_id}) when is_binary(user_id) do
    {:ok, %{"user_id" => user_id}}
  end

  defp validate_dm_params(_), do: {:error, "invalid_params"}

  defp validate_push_token_params(%{"token" => token, "platform" => platform})
       when is_binary(token) and is_binary(platform) do
    {:ok, %{"token" => token, "platform" => platform}}
  end

  defp validate_push_token_params(_), do: {:error, "invalid_params"}

  defp validate_profile_params(params) when is_map(params) do
    allowed_fields = ["username", "avatar_url"]
    filtered_params = Map.take(params, allowed_fields)
    {:ok, filtered_params}
  end

  defp validate_profile_params(_), do: {:error, "invalid_params"}

  defp update_user_profile(user, params) do
    changeset = User.changeset(user, params)
    Sup.Repo.update(changeset)
  end

  defp validate_autocomplete_params(params) when is_map(params) do
    case Map.get(params, "text") do
      text when is_binary(text) ->
        result = %{"text" => text}

        # Add optional parameters if present
        result =
          if is_binary(Map.get(params, "room_id")) do
            Map.put(result, "room_id", Map.get(params, "room_id"))
          else
            result
          end

        result =
          if is_integer(Map.get(params, "limit")) do
            Map.put(result, "limit", Map.get(params, "limit"))
          else
            result
          end

        {:ok, result}

      _ ->
        {:error, "invalid_params"}
    end
  end

  defp validate_autocomplete_params(_), do: {:error, "invalid_params"}

  defp validate_completion_params(params) when is_map(params) do
    case Map.get(params, "text") do
      text when is_binary(text) ->
        result = %{"text" => text}

        # Add optional parameters if present
        result =
          if is_binary(Map.get(params, "room_id")) do
            Map.put(result, "room_id", Map.get(params, "room_id"))
          else
            result
          end

        result =
          if is_integer(Map.get(params, "max_length")) do
            Map.put(result, "max_length", Map.get(params, "max_length"))
          else
            result
          end

        {:ok, result}

      _ ->
        {:error, "invalid_params"}
    end
  end

  defp validate_completion_params(_), do: {:error, "invalid_params"}

  # Friend management validation helpers
  defp validate_friend_request_params(%{"target_user_id" => target_user_id})
       when is_binary(target_user_id) do
    {:ok, %{"target_user_id" => target_user_id}}
  end

  defp validate_friend_request_params(_), do: {:error, "invalid_params"}

  defp validate_friend_response_params(%{"action" => action})
       when action in ["accept", "reject"] do
    {:ok, %{"action" => action}}
  end

  defp validate_friend_response_params(_), do: {:error, "invalid_params"}

  defp validate_block_user_params(%{"target_user_id" => target_user_id})
       when is_binary(target_user_id) do
    {:ok, %{"target_user_id" => target_user_id}}
  end

  defp validate_block_user_params(_), do: {:error, "invalid_params"}

  # Call validation helpers
  defp validate_call_params(%{"target_user_id" => target_user_id, "call_type" => call_type})
       when is_binary(target_user_id) and call_type in ["voice", "video"] do
    {:ok, %{"target_user_id" => target_user_id, "call_type" => call_type}}
  end

  defp validate_call_params(_), do: {:error, "invalid_params"}

  defp validate_signal_params(%{"signal" => signal}) when is_map(signal) do
    {:ok, %{"signal" => signal}}
  end

  defp validate_signal_params(_), do: {:error, "invalid_params"}

  # Settings validation helpers
  defp validate_settings_params(params) when is_map(params) do
    allowed_fields = [
      "display_name",
      "bio",
      "status_message",
      "theme_preference",
      "accent_color",
      "activity_status",
      "notification_settings",
      "privacy_settings",
      "call_settings"
    ]

    filtered_params = Map.take(params, allowed_fields)
    {:ok, filtered_params}
  end

  defp validate_settings_params(_), do: {:error, "invalid_params"}

  defp update_user_settings(user, params) do
    changeset = User.changeset(user, params)
    Sup.Repo.update(changeset)
  end
end
