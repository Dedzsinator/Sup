defmodule Sup.WebSocket.Handler do
  @moduledoc """
  WebSocket handler for real-time messaging using websock behavior.
  Handles connection lifecycle, message routing, and presence management.
  """

  @behaviour WebSock
  require Logger

  alias Sup.Messaging.{MessageService, EnhancedMessageService}
  alias Sup.Presence.PresenceService
  alias Sup.Room.RoomService

  defstruct [:user_id, :connection_id, :subscriptions]

  @impl WebSock
  def init(%{user_id: user_id}) do
    connection_id = generate_connection_id()

    # Register connection in registry
    Registry.register(Sup.ConnectionRegistry, user_id, connection_id)

    # Set user online presence
    PresenceService.user_online(user_id, connection_id)

    # Subscribe to user's personal channel
    Phoenix.PubSub.subscribe(Sup.PubSub, "user:#{user_id}")

    # Get user's rooms and subscribe to them
    rooms = RoomService.get_user_rooms(user_id)

    subscriptions =
      Enum.map(rooms, fn room ->
        Phoenix.PubSub.subscribe(Sup.PubSub, "room:#{room.id}")
        room.id
      end)

    state = %__MODULE__{
      user_id: user_id,
      connection_id: connection_id,
      subscriptions: subscriptions
    }

    Logger.info("WebSocket connected for user #{user_id}")
    {:ok, state}
  end

  @impl WebSock
  def handle_in({text, [opcode: :text]}, state) do
    case Jason.decode(text) do
      {:ok, message} ->
        handle_message(message, state)

      {:error, _} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: "invalid_json"
          })

        {:reply, {:text, error_response}, state}
    end
  end

  @impl WebSock
  def handle_info({:message, message}, state) do
    response =
      Jason.encode!(%{
        type: "message",
        data: message
      })

    {:reply, {:text, response}, state}
  end

  def handle_info({:typing, data}, state) do
    response =
      Jason.encode!(%{
        type: "typing",
        data: data
      })

    {:reply, {:text, response}, state}
  end

  def handle_info({:presence, data}, state) do
    response =
      Jason.encode!(%{
        type: "presence",
        data: data
      })

    {:reply, {:text, response}, state}
  end

  def handle_info({:delivery_receipt, data}, state) do
    response =
      Jason.encode!(%{
        type: "delivery_receipt",
        data: data
      })

    {:reply, {:text, response}, state}
  end

  @impl WebSock
  def terminate(reason, state) do
    Logger.info("WebSocket disconnected for user #{state.user_id}, reason: #{inspect(reason)}")

    # Set user offline
    PresenceService.user_offline(state.user_id, state.connection_id)

    # Unregister connection
    Registry.unregister(Sup.ConnectionRegistry, state.user_id)

    :ok
  end

  # Message handlers
  defp handle_message(%{"type" => "send_message", "data" => data}, state) do
    case EnhancedMessageService.send_message(state.user_id, data) do
      {:ok, message} ->
        # Optimistic response to sender
        response =
          Jason.encode!(%{
            type: "message_sent",
            data: message
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "edit_message", "data" => %{"message_id" => message_id, "content" => content}}, state) do
    case EnhancedMessageService.edit_message(message_id, state.user_id, content) do
      {:ok, message} ->
        response =
          Jason.encode!(%{
            type: "message_edited",
            data: message
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "delete_message", "data" => %{"message_id" => message_id}}, state) do
    case EnhancedMessageService.delete_message(message_id, state.user_id) do
      :ok ->
        response =
          Jason.encode!(%{
            type: "message_deleted",
            data: %{message_id: message_id}
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  # Reaction handlers
  defp handle_message(%{"type" => "add_reaction", "data" => %{"message_id" => message_id, "emoji" => emoji}}, state) do
    case EnhancedMessageService.add_reaction(message_id, state.user_id, emoji) do
      {:ok, reaction} ->
        response =
          Jason.encode!(%{
            type: "reaction_added",
            data: %{message_id: message_id, reaction: reaction}
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "remove_reaction", "data" => %{"message_id" => message_id, "emoji" => emoji}}, state) do
    case EnhancedMessageService.remove_reaction(message_id, state.user_id, emoji) do
      :ok ->
        response =
          Jason.encode!(%{
            type: "reaction_removed",
            data: %{message_id: message_id, emoji: emoji, user_id: state.user_id}
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  # Thread handlers
  defp handle_message(%{"type" => "create_thread", "data" => %{"message_id" => message_id, "initial_reply" => initial_reply}}, state) do
    case EnhancedMessageService.create_thread(message_id, state.user_id, initial_reply) do
      {:ok, thread} ->
        response =
          Jason.encode!(%{
            type: "thread_created",
            data: thread
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "thread_reply", "data" => %{"thread_id" => thread_id, "content" => content, "type" => type}}, state) do
    params = %{"content" => content, "type" => type}
    case EnhancedMessageService.reply_to_thread(thread_id, state.user_id, params) do
      {:ok, message} ->
        response =
          Jason.encode!(%{
            type: "thread_message_sent",
            data: message
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  # Custom emoji handlers
  defp handle_message(%{"type" => "create_custom_emoji", "data" => %{"room_id" => room_id, "name" => name, "image_url" => image_url} = data}, state) do
    tags = Map.get(data, "tags", [])
    params = %{"name" => name, "image_url" => image_url, "tags" => tags}
    
    case Sup.Messaging.CustomEmojiService.create_emoji(room_id, state.user_id, params) do
      {:ok, emoji} ->
        response =
          Jason.encode!(%{
            type: "custom_emoji_added",
            data: emoji
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "delete_custom_emoji", "data" => %{"room_id" => room_id, "emoji_id" => emoji_id}}, state) do
    case Sup.Messaging.CustomEmojiService.delete_emoji(emoji_id, state.user_id) do
      :ok ->
        response =
          Jason.encode!(%{
            type: "custom_emoji_deleted",
            data: %{room_id: room_id, emoji_id: emoji_id}
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  # Search handlers
  defp handle_message(%{"type" => "search_messages", "data" => %{"query" => query} = data}, state) do
    room_id = Map.get(data, "room_id")
    limit = Map.get(data, "limit", 20)
    
    case MessageService.search_messages(state.user_id, query, limit) do
      {:ok, messages} ->
        response =
          Jason.encode!(%{
            type: "search_results",
            data: %{query: query, messages: messages}
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  # Voice/Video call handlers
  defp handle_message(%{"type" => "initiate_call", "data" => %{"room_id" => room_id, "type" => call_type, "participants" => participants}}, state) do
    case Sup.Voice.CallService.initiate_call(state.user_id, participants, call_type) do
      {:ok, call} ->
        response =
          Jason.encode!(%{
            type: "call_initiated",
            data: call
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "answer_call", "data" => %{"call_id" => call_id}}, state) do
    case Sup.Voice.CallService.accept_call(call_id, state.user_id) do
      {:ok, call} ->
        response =
          Jason.encode!(%{
            type: "call_answered",
            data: call
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "decline_call", "data" => %{"call_id" => call_id}}, state) do
    case Sup.Voice.CallService.reject_call(call_id, state.user_id) do
      {:ok, call} ->
        response =
          Jason.encode!(%{
            type: "call_declined",
            data: call
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "end_call", "data" => %{"call_id" => call_id}}, state) do
    case Sup.Voice.CallService.end_call(call_id, state.user_id) do
      {:ok, call} ->
        response =
          Jason.encode!(%{
            type: "call_ended",
            data: call
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "webrtc_signal", "data" => %{"call_id" => call_id, "signal" => signal}}, state) do
    # Forward WebRTC signaling data
    response =
      Jason.encode!(%{
        type: "webrtc_signaling",
        data: %{call_id: call_id, signal: signal, from: state.user_id}
      })

    # Broadcast to call participants
    Phoenix.PubSub.broadcast(Sup.PubSub, "call:#{call_id}", {:webrtc_signal, signal, state.user_id})
    
    {:reply, {:text, response}, state}
  end

  # Multi-device sync handlers
  defp handle_message(%{"type" => "request_sync", "data" => _data}, state) do
    case Sup.Messaging.MultiDeviceSyncService.get_device_state(state.user_id) do
      {:ok, sync_state} ->
        response =
          Jason.encode!(%{
            type: "sync_state_updated",
            data: sync_state
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "sync_device_state", "data" => device_info}, state) do
    case Sup.Messaging.MultiDeviceSyncService.sync_device_state(state.user_id, device_info) do
      {:ok, sync_result} ->
        response =
          Jason.encode!(%{
            type: "device_state_synced",
            data: sync_result
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  # Offline message handlers
  defp handle_message(%{"type" => "request_offline_messages", "data" => _data}, state) do
    case Sup.Messaging.OfflineQueueService.get_offline_messages(state.user_id) do
      {:ok, messages} ->
        response =
          Jason.encode!(%{
            type: "offline_messages",
            data: %{messages: messages}
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "acknowledge_missed_messages", "data" => %{"message_ids" => message_ids}}, state) do
    case Sup.Messaging.OfflineQueueService.mark_messages_received(state.user_id, message_ids) do
      :ok ->
        response =
          Jason.encode!(%{
            type: "messages_acknowledged",
            data: %{message_ids: message_ids}
          })

        {:reply, {:text, response}, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  # Presence and activity handlers
  defp handle_message(%{"type" => "update_presence", "data" => %{"status" => status}}, state) do
    PresenceService.update_user_status(state.user_id, status)
    
    response =
      Jason.encode!(%{
        type: "presence_updated",
        data: %{user_id: state.user_id, status: status}
      })

    {:reply, {:text, response}, state}
  end

  defp handle_message(%{"type" => "update_activity", "data" => activity}, state) do
    PresenceService.update_user_activity(state.user_id, activity)
    
    response =
      Jason.encode!(%{
        type: "activity_updated",
        data: %{user_id: state.user_id, activity: activity}
      })

    {:reply, {:text, response}, state}
  end

  defp handle_message(%{"type" => "typing_start", "data" => %{"room_id" => room_id}}, state) do
    PresenceService.user_typing(state.user_id, room_id, true)
    {:ok, state}
  end

  defp handle_message(%{"type" => "typing_stop", "data" => %{"room_id" => room_id}}, state) do
    PresenceService.user_typing(state.user_id, room_id, false)
    {:ok, state}
  end

  defp handle_message(%{"type" => "mark_read", "data" => %{"message_id" => message_id}}, state) do
    MessageService.mark_message_read(message_id, state.user_id)
    {:ok, state}
  end

  defp handle_message(%{"type" => "join_room", "data" => %{"room_id" => room_id}}, state) do
    case RoomService.can_join_room?(state.user_id, room_id) do
      true ->
        Phoenix.PubSub.subscribe(Sup.PubSub, "room:#{room_id}")
        new_subscriptions = [room_id | state.subscriptions]
        {:ok, %{state | subscriptions: new_subscriptions}}

      false ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: "unauthorized_room_access"
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(message, state) do
    Logger.warn("Unknown message type: #{inspect(message)}")

    error_response =
      Jason.encode!(%{
        type: "error",
        error: "unknown_message_type"
      })

    {:reply, {:text, error_response}, state}
  end

  defp generate_connection_id do
    :crypto.strong_rand_bytes(16) |> Base.encode64()
  end
end
