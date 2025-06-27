defmodule Sup.WebSocket.EnhancedHandler do
  @moduledoc """
  Enhanced WebSocket handler supporting E2E encryption, message threading,
  reactions, enhanced presence, voice/video calls, and real-time features.
  """

  @behaviour WebSock
  require Logger

  alias Sup.Messaging.EnhancedMessageService
  alias Sup.Presence.EnhancedPresenceService
  alias Sup.Room.RoomService
  alias Sup.Voice.CallService
  alias Sup.Security.RateLimit

  defstruct [:user_id, :connection_id, :subscriptions, :device_info, :rate_limiter]

  @impl WebSock
  def init(%{user_id: user_id}) do
    connection_id = generate_connection_id()

    # Register connection in registry
    Registry.register(Sup.ConnectionRegistry, user_id, connection_id)

    # Set user online presence
    EnhancedPresenceService.set_user_presence(user_id, :online)

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
      subscriptions: subscriptions,
      device_info: get_device_info(),
      # Rate limiter will use the existing RateLimit module
      rate_limiter: nil
    }

    Logger.info("Enhanced WebSocket connected for user #{user_id}")
    {:ok, state}
  end

  @impl WebSock
  def handle_in({text, [opcode: :text]}, state) do
    # Check rate limits using existing RateLimit module
    socket_stub = %{assigns: %{user_id: state.user_id}}

    case RateLimit.check_rate_limit(socket_stub, :websocket) do
      :ok ->
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

      {:error, :rate_limit_exceeded} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: "rate_limit_exceeded"
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

  def handle_info({:reaction, reaction_data}, state) do
    response =
      Jason.encode!(%{
        type: "reaction",
        data: reaction_data
      })

    {:reply, {:text, response}, state}
  end

  def handle_info({:message_edited, edit_data}, state) do
    response =
      Jason.encode!(%{
        type: "message_edited",
        data: edit_data
      })

    {:reply, {:text, response}, state}
  end

  def handle_info({:message_deleted, delete_data}, state) do
    response =
      Jason.encode!(%{
        type: "message_deleted",
        data: delete_data
      })

    {:reply, {:text, response}, state}
  end

  def handle_info({:presence_update, presence_data}, state) do
    response =
      Jason.encode!(%{
        type: "presence_update",
        data: presence_data
      })

    {:reply, {:text, response}, state}
  end

  def handle_info({:activity_update, activity_data}, state) do
    response =
      Jason.encode!(%{
        type: "activity_update",
        data: activity_data
      })

    {:reply, {:text, response}, state}
  end

  def handle_info({:voice_presence_update, voice_data}, state) do
    response =
      Jason.encode!(%{
        type: "voice_presence_update",
        data: voice_data
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

  def handle_info({:call_event, call_data}, state) do
    response =
      Jason.encode!(%{
        type: "call_event",
        data: call_data
      })

    {:reply, {:text, response}, state}
  end

  def handle_info({:webrtc_signal, signal_data}, state) do
    response =
      Jason.encode!(%{
        type: "webrtc_signal",
        data: signal_data
      })

    {:reply, {:text, response}, state}
  end

  @impl WebSock
  def terminate(reason, state) do
    Logger.info(
      "Enhanced WebSocket disconnected for user #{state.user_id}, reason: #{inspect(reason)}"
    )

    # Clear user activities
    Enum.each(state.subscriptions, fn room_id ->
      EnhancedPresenceService.clear_user_activity(state.user_id, room_id, :typing)
      EnhancedPresenceService.clear_user_activity(state.user_id, room_id, :recording_audio)
      EnhancedPresenceService.clear_user_activity(state.user_id, room_id, :recording_video)
    end)

    # Set user offline if no other connections
    case Registry.lookup(Sup.ConnectionRegistry, state.user_id) do
      [_single_connection] ->
        EnhancedPresenceService.set_user_presence(state.user_id, :offline)

      _multiple_connections ->
        :ok
    end

    # Unregister connection
    Registry.unregister(Sup.ConnectionRegistry, state.user_id)

    :ok
  end

  # Message handlers
  defp handle_message(%{"type" => "send_message", "data" => data}, state) do
    # Enhanced message sending with E2E encryption option
    case data["encrypted"] do
      true ->
        handle_encrypted_message_send(data, state)

      _ ->
        handle_regular_message_send(data, state)
    end
  end

  defp handle_message(%{"type" => "send_thread_message", "data" => data}, state) do
    case EnhancedMessageService.create_thread(state.user_id, data["parent_message_id"], data) do
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

  defp handle_message(
         %{"type" => "add_reaction", "data" => %{"message_id" => message_id, "emoji" => emoji}},
         state
       ) do
    case EnhancedMessageService.add_reaction(state.user_id, message_id, emoji) do
      {:ok, _reaction} ->
        {:ok, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(
         %{
           "type" => "remove_reaction",
           "data" => %{"message_id" => message_id, "emoji" => emoji}
         },
         state
       ) do
    case EnhancedMessageService.remove_reaction(state.user_id, message_id, emoji) do
      {:ok, _reaction} ->
        {:ok, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(
         %{
           "type" => "edit_message",
           "data" => %{"message_id" => message_id, "content" => content}
         },
         state
       ) do
    case EnhancedMessageService.edit_message(state.user_id, message_id, content) do
      {:ok, result} ->
        response =
          Jason.encode!(%{
            type: "message_edit_confirmed",
            data: result
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

  defp handle_message(
         %{"type" => "delete_message", "data" => %{"message_id" => message_id}},
         state
       ) do
    case EnhancedMessageService.delete_message(state.user_id, message_id) do
      {:ok, _} ->
        {:ok, state}

      {:error, reason} ->
        error_response =
          Jason.encode!(%{
            type: "error",
            error: reason
          })

        {:reply, {:text, error_response}, state}
    end
  end

  defp handle_message(%{"type" => "set_presence", "data" => %{"status" => status} = data}, state) do
    custom_message = Map.get(data, "custom_message")
    status_atom = String.to_existing_atom(status)

    EnhancedPresenceService.set_user_presence(state.user_id, status_atom, custom_message)
    {:ok, state}
  end

  defp handle_message(%{"type" => "start_activity", "data" => data}, state) do
    activity_type = String.to_existing_atom(data["activity_type"])
    room_id = data["room_id"]
    metadata = Map.get(data, "metadata", %{})

    EnhancedPresenceService.set_user_activity(state.user_id, room_id, activity_type, metadata)
    {:ok, state}
  end

  defp handle_message(%{"type" => "stop_activity", "data" => data}, state) do
    activity_type = String.to_existing_atom(data["activity_type"])
    room_id = data["room_id"]

    EnhancedPresenceService.clear_user_activity(state.user_id, room_id, activity_type)
    {:ok, state}
  end

  defp handle_message(%{"type" => "typing_start", "data" => %{"room_id" => room_id}}, state) do
    EnhancedPresenceService.set_user_activity(state.user_id, room_id, :typing)
    {:ok, state}
  end

  defp handle_message(%{"type" => "typing_stop", "data" => %{"room_id" => room_id}}, state) do
    EnhancedPresenceService.clear_user_activity(state.user_id, room_id, :typing)
    {:ok, state}
  end

  defp handle_message(%{"type" => "mark_read", "data" => %{"message_id" => message_id}}, state) do
    EnhancedMessageService.mark_message_read(message_id, state.user_id)
    {:ok, state}
  end

  defp handle_message(
         %{"type" => "mark_thread_read", "data" => %{"thread_id" => thread_id}},
         state
       ) do
    EnhancedMessageService.mark_thread_read(thread_id, state.user_id)
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

  # Voice/Video Call handlers
  defp handle_message(%{"type" => "initiate_call", "data" => data}, state) do
    case CallService.initiate_call(state.user_id, data) do
      {:ok, call} ->
        # Set voice presence
        EnhancedPresenceService.set_voice_presence(
          state.user_id,
          call.room_id,
          call.id,
          String.to_existing_atom(call.call_type)
        )

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
    case CallService.accept_call(call_id, state.user_id) do
      {:ok, call} ->
        # Set voice presence
        EnhancedPresenceService.set_voice_presence(
          state.user_id,
          call.room_id,
          call.id,
          String.to_existing_atom(call.call_type)
        )

        {:ok, state}

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
    CallService.end_call(call_id, state.user_id)
    EnhancedPresenceService.clear_voice_presence(state.user_id)
    {:ok, state}
  end

  defp handle_message(%{"type" => "webrtc_signal", "data" => data}, state) do
    CallService.handle_webrtc_signal(data["call_id"], state.user_id, data)
    {:ok, state}
  end

  # Search and discovery
  defp handle_message(%{"type" => "search_messages", "data" => data}, state) do
    case EnhancedMessageService.search_messages(state.user_id, data["query"], data) do
      {:ok, results} ->
        response =
          Jason.encode!(%{
            type: "search_results",
            data: %{
              query: data["query"],
              results: results
            }
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

  defp handle_message(%{"type" => "get_mentions", "data" => _data}, state) do
    case EnhancedMessageService.search_mentions(state.user_id) do
      {:ok, mentions} ->
        response =
          Jason.encode!(%{
            type: "mentions",
            data: mentions
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

  # Unknown message handler
  defp handle_message(message, state) do
    Logger.warning("Unknown message type: #{inspect(message)}")

    error_response =
      Jason.encode!(%{
        type: "error",
        error: "unknown_message_type"
      })

    {:reply, {:text, error_response}, state}
  end

  # Private helper functions
  defp handle_encrypted_message_send(data, state) do
    case EnhancedMessageService.send_encrypted_message(state.user_id, data) do
      {:ok, message} ->
        response =
          Jason.encode!(%{
            type: "message_sent",
            data: Map.put(message, :encrypted, true)
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

  defp handle_regular_message_send(data, state) do
    case EnhancedMessageService.send_message(state.user_id, data) do
      {:ok, message} ->
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

  defp get_device_info do
    %{
      # Could be determined from user agent
      platform: "web",
      version: "1.0.0",
      connected_at: DateTime.utc_now()
    }
  end

  defp generate_connection_id do
    :crypto.strong_rand_bytes(16) |> Base.encode64()
  end
end
