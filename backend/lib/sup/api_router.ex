defmodule Sup.ApiRouter do
  @moduledoc """
  Protected API routes requiring authentication.
  """

  use Plug.Router
  require Logger

  alias Sup.Auth.{Guardian, User, FriendService}
  alias Sup.Room.RoomService

  alias Sup.Messaging.{
    MessageService,
    EnhancedMessageService,
    CustomEmojiService,
    OfflineQueueService,
    AnalyticsService
  }

  alias Sup.Sync.MultiDeviceSyncService

  alias Sup.Voice.CallService
  alias Sup.Autocomplete.Service, as: AutocompleteService
  alias Sup.Security.{RateLimitPlug, AuthorizationPlug}
  alias Sup.SpamDetection.Service, as: SpamDetectionService

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

  post "/upload/media" do
    # Handle rich media uploads
    send_resp(conn, 501, Jason.encode!(%{error: "not_implemented"}))
  end

  # Message Management Endpoints
  post "/messages" do
    with {:ok, params} <- validate_send_message_params(conn.body_params),
         {:ok, message} <-
           EnhancedMessageService.send_message(conn.assigns.current_user.id, params) do
      send_resp(conn, 201, Jason.encode!(%{message: message}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  put "/messages/:message_id" do
    message_id = conn.path_params["message_id"]
    user_id = conn.assigns.current_user.id

    with {:ok, params} <- validate_edit_message_params(conn.body_params),
         {:ok, message} <-
           EnhancedMessageService.edit_message(message_id, user_id, params["content"]) do
      send_resp(conn, 200, Jason.encode!(%{message: message}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  delete "/messages/:message_id" do
    message_id = conn.path_params["message_id"]
    user_id = conn.assigns.current_user.id

    case EnhancedMessageService.delete_message(message_id, user_id) do
      :ok ->
        send_resp(conn, 200, Jason.encode!(%{message: "message_deleted"}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/messages/search" do
    query = conn.query_params["q"]
    _room_id = conn.query_params["room_id"]
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

  # Message Reactions
  post "/messages/:message_id/reactions" do
    message_id = conn.path_params["message_id"]
    user_id = conn.assigns.current_user.id

    with {:ok, params} <- validate_reaction_params(conn.body_params),
         {:ok, reaction} <-
           EnhancedMessageService.add_reaction(message_id, user_id, params["emoji"]) do
      send_resp(conn, 201, Jason.encode!(%{reaction: reaction}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  delete "/messages/:message_id/reactions/:emoji" do
    message_id = conn.path_params["message_id"]
    emoji = conn.path_params["emoji"]
    user_id = conn.assigns.current_user.id

    case EnhancedMessageService.remove_reaction(message_id, user_id, emoji) do
      :ok ->
        send_resp(conn, 200, Jason.encode!(%{message: "reaction_removed"}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/messages/:message_id/reactions" do
    message_id = conn.path_params["message_id"]

    case EnhancedMessageService.get_message_reactions(message_id) do
      {:ok, reactions} ->
        send_resp(conn, 200, Jason.encode!(%{reactions: reactions}))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  # Message Threads
  get "/messages/:message_id/thread" do
    message_id = conn.path_params["message_id"]

    case EnhancedMessageService.get_thread(message_id) do
      {:ok, thread} ->
        send_resp(conn, 200, Jason.encode!(thread))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/messages/:message_id/thread/messages" do
    message_id = conn.path_params["message_id"]
    limit = String.to_integer(conn.query_params["limit"] || "50")
    before = conn.query_params["before"]

    case EnhancedMessageService.get_thread_messages(message_id, limit, before) do
      {:ok, messages} ->
        send_resp(conn, 200, Jason.encode!(%{messages: messages}))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  post "/messages/:message_id/thread/reply" do
    message_id = conn.path_params["message_id"]
    user_id = conn.assigns.current_user.id

    with {:ok, params} <- validate_thread_reply_params(conn.body_params),
         {:ok, message} <- EnhancedMessageService.reply_to_thread(message_id, user_id, params) do
      send_resp(conn, 201, Jason.encode!(%{message: message}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  # Custom Emojis
  get "/rooms/:room_id/emojis" do
    room_id = conn.path_params["room_id"]

    case CustomEmojiService.get_room_emojis(room_id) do
      {:ok, emojis} ->
        send_resp(conn, 200, Jason.encode!(%{emojis: emojis}))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  get "/emojis/global" do
    case CustomEmojiService.get_global_emojis() do
      {:ok, emojis} ->
        send_resp(conn, 200, Jason.encode!(%{emojis: emojis}))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  post "/rooms/:room_id/emojis" do
    room_id = conn.path_params["room_id"]
    user_id = conn.assigns.current_user.id

    with {:ok, params} <- validate_custom_emoji_params(conn.body_params),
         {:ok, emoji} <- CustomEmojiService.create_emoji(room_id, user_id, params) do
      send_resp(conn, 201, Jason.encode!(emoji))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  delete "/rooms/:room_id/emojis/:emoji_id" do
    _room_id = conn.path_params["room_id"]
    emoji_id = conn.path_params["emoji_id"]
    user_id = conn.assigns.current_user.id

    case CustomEmojiService.delete_emoji(emoji_id, user_id) do
      :ok ->
        send_resp(conn, 200, Jason.encode!(%{message: "emoji_deleted"}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/emojis/search" do
    query = conn.query_params["q"]
    room_id = conn.query_params["room_id"]

    if query && String.length(query) >= 2 do
      case CustomEmojiService.search_emojis(query, room_id) do
        {:ok, emojis} ->
          send_resp(conn, 200, Jason.encode!(%{emojis: emojis}))

        {:error, reason} ->
          send_resp(conn, 500, Jason.encode!(%{error: reason}))
      end
    else
      send_resp(conn, 400, Jason.encode!(%{error: "query_too_short"}))
    end
  end

  # Offline Messages
  get "/messages/offline" do
    user_id = conn.assigns.current_user.id

    case OfflineQueueService.get_offline_messages(user_id) do
      {:ok, messages} ->
        send_resp(conn, 200, Jason.encode!(%{messages: messages}))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  post "/messages/offline/received" do
    user_id = conn.assigns.current_user.id

    with {:ok, params} <- validate_offline_received_params(conn.body_params),
         :ok <- OfflineQueueService.mark_messages_received(user_id, params["message_ids"]) do
      send_resp(conn, 200, Jason.encode!(%{message: "messages_marked_received"}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  # Analytics
  get "/analytics/messages" do
    user_id = conn.assigns.current_user.id
    room_id = conn.query_params["room_id"]
    period = conn.query_params["period"] || "7d"

    case AnalyticsService.get_message_analytics(user_id, room_id, period) do
      {:ok, analytics} ->
        send_resp(conn, 200, Jason.encode!(analytics))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  get "/rooms/:room_id/insights" do
    room_id = conn.path_params["room_id"]
    user_id = conn.assigns.current_user.id

    # Check if user can access this room
    if RoomService.can_send_message?(user_id, room_id) do
      case AnalyticsService.get_room_insights(room_id) do
        {:ok, insights} ->
          send_resp(conn, 200, Jason.encode!(insights))

        {:error, reason} ->
          send_resp(conn, 500, Jason.encode!(%{error: reason}))
      end
    else
      send_resp(conn, 403, Jason.encode!(%{error: "unauthorized"}))
    end
  end

  # Multi-device Sync
  get "/sync/device-state" do
    user_id = conn.assigns.current_user.id

    case MultiDeviceSyncService.get_device_state(user_id) do
      {:ok, state} ->
        send_resp(conn, 200, Jason.encode!(state))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  post "/sync/register-device" do
    user_id = conn.assigns.current_user.id

    with {:ok, params} <- validate_device_params(conn.body_params),
         {:ok, device} <- MultiDeviceSyncService.register_device(user_id, params) do
      send_resp(conn, 201, Jason.encode!(device))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/sync/devices" do
    user_id = conn.assigns.current_user.id

    case MultiDeviceSyncService.get_user_devices(user_id) do
      {:ok, devices} ->
        send_resp(conn, 200, Jason.encode!(%{devices: devices}))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: reason}))
    end
  end

  delete "/sync/devices/:device_id" do
    device_id = conn.path_params["device_id"]
    user_id = conn.assigns.current_user.id

    case MultiDeviceSyncService.remove_device(user_id, device_id) do
      :ok ->
        send_resp(conn, 200, Jason.encode!(%{message: "device_removed"}))

      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
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

  # Advanced messaging validation helpers
  defp validate_send_message_params(
         params = %{"room_id" => room_id, "content" => content, "type" => type}
       )
       when is_binary(room_id) and is_binary(content) and is_binary(type) do
    base_params = %{"room_id" => room_id, "content" => content, "type" => type}

    # Add optional reply_to_id if present
    final_params =
      case Map.get(params, "reply_to_id") do
        reply_to_id when is_binary(reply_to_id) ->
          Map.put(base_params, "reply_to_id", reply_to_id)

        _ ->
          base_params
      end

    {:ok, final_params}
  end

  defp validate_send_message_params(_), do: {:error, "invalid_params"}

  defp validate_edit_message_params(%{"content" => content}) when is_binary(content) do
    {:ok, %{"content" => content}}
  end

  defp validate_edit_message_params(_), do: {:error, "invalid_params"}

  defp validate_reaction_params(%{"emoji" => emoji}) when is_binary(emoji) do
    {:ok, %{"emoji" => emoji}}
  end

  defp validate_reaction_params(_), do: {:error, "invalid_params"}

  defp validate_thread_reply_params(%{"content" => content, "type" => type})
       when is_binary(content) and is_binary(type) do
    {:ok, %{"content" => content, "type" => type}}
  end

  defp validate_thread_reply_params(_), do: {:error, "invalid_params"}

  defp validate_custom_emoji_params(params = %{"name" => name, "image_url" => image_url})
       when is_binary(name) and is_binary(image_url) do
    base_params = %{"name" => name, "image_url" => image_url}

    # Add optional tags if present
    final_params =
      case Map.get(params, "tags") do
        tags when is_list(tags) -> Map.put(base_params, "tags", tags)
        _ -> base_params
      end

    {:ok, final_params}
  end

  defp validate_custom_emoji_params(_), do: {:error, "invalid_params"}

  defp validate_offline_received_params(%{"message_ids" => message_ids})
       when is_list(message_ids) do
    {:ok, %{"message_ids" => message_ids}}
  end

  defp validate_offline_received_params(_), do: {:error, "invalid_params"}

  defp validate_device_params(
         params = %{"device_name" => device_name, "device_type" => device_type}
       )
       when is_binary(device_name) and is_binary(device_type) do
    base_params = %{"device_name" => device_name, "device_type" => device_type}

    # Add optional device_info if present
    final_params =
      case Map.get(params, "device_info") do
        device_info when is_map(device_info) -> Map.put(base_params, "device_info", device_info)
        _ -> base_params
      end

    {:ok, final_params}
  end

  defp validate_device_params(_), do: {:error, "invalid_params"}

  # Spam Detection endpoints

  post "/spam/check" do
    with {:ok, params} <- validate_spam_check_params(conn.body_params) do
      user_id = conn.assigns.current_user.id
      message = Map.get(params, "message")

      result = SpamDetectionService.check_message_only(message, user_id)

      send_resp(conn, 200, Jason.encode!(%{spam_check: result}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  post "/spam/report" do
    with {:ok, params} <- validate_spam_report_params(conn.body_params) do
      user_id = conn.assigns.current_user.id
      message = Map.get(params, "message")
      is_spam = Map.get(params, "is_spam")

      case SpamDetectionService.report_message(message, user_id, is_spam) do
        :ok ->
          send_resp(conn, 200, Jason.encode!(%{status: "reported"}))

        {:error, reason} ->
          send_resp(conn, 500, Jason.encode!(%{error: "Failed to report: #{inspect(reason)}"}))
      end
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/spam/stats" do
    case SpamDetectionService.get_stats() do
      {:ok, stats} ->
        send_resp(conn, 200, Jason.encode!(%{stats: stats}))

      {:error, reason} ->
        send_resp(conn, 500, Jason.encode!(%{error: "Failed to get stats: #{inspect(reason)}"}))
    end
  end

  get "/spam/health" do
    is_healthy = SpamDetectionService.health_check()

    status_code = if is_healthy, do: 200, else: 503
    status = if is_healthy, do: "healthy", else: "unhealthy"

    send_resp(conn, status_code, Jason.encode!(%{status: status, service: "spam_detection"}))
  end

  # Validation helpers for spam detection

  defp validate_spam_check_params(params) when is_map(params) do
    with message when is_binary(message) and byte_size(message) > 0 <- Map.get(params, "message") do
      {:ok, %{"message" => message}}
    else
      _ -> {:error, "message_required"}
    end
  end

  defp validate_spam_check_params(_), do: {:error, "invalid_params"}

  defp validate_spam_report_params(params) when is_map(params) do
    with message when is_binary(message) and byte_size(message) > 0 <- Map.get(params, "message"),
         is_spam when is_boolean(is_spam) <- Map.get(params, "is_spam") do
      {:ok, %{"message" => message, "is_spam" => is_spam}}
    else
      _ -> {:error, "message_and_is_spam_required"}
    end
  end

  defp validate_spam_report_params(_), do: {:error, "invalid_params"}
end
