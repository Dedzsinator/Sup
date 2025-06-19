defmodule Sup.WebSocket.Handler do
  @moduledoc """
  WebSocket handler for real-time messaging using websock behavior.
  Handles connection lifecycle, message routing, and presence management.
  """

  @behaviour WebSock
  require Logger

  alias Sup.Messaging.MessageService
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
    case MessageService.send_message(state.user_id, data) do
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
