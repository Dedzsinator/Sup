defmodule Sup.ApiRouter do
  @moduledoc """
  Protected API routes requiring authentication.
  """

  use Plug.Router
  require Logger

  alias Sup.Auth.{Guardian, User}
  alias Sup.Room.RoomService
  alias Sup.Messaging.MessageService
  alias Sup.Autocomplete.Service, as: AutocompleteService

  plug(:match)
  plug(:authenticate)
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
         {:ok, suggestions} <- AutocompleteService.get_suggestions(
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
         {:ok, completion} <- AutocompleteService.get_completion(
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
        result = if is_binary(Map.get(params, "room_id")) do
          Map.put(result, "room_id", Map.get(params, "room_id"))
        else
          result
        end
        
        result = if is_integer(Map.get(params, "limit")) do
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
        result = if is_binary(Map.get(params, "room_id")) do
          Map.put(result, "room_id", Map.get(params, "room_id"))
        else
          result
        end
        
        result = if is_integer(Map.get(params, "max_length")) do
          Map.put(result, "max_length", Map.get(params, "max_length"))
        else
          result
        end
        
        {:ok, result}
      
      _ ->
        {:error, "invalid_params"}
    end
  end
end
