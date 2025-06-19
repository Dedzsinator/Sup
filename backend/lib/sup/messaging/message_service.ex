defmodule Sup.Messaging.MessageService do
  @moduledoc """
  Core messaging service handling message sending, delivery, and routing.
  """

  alias Sup.Messaging.{Message, DeliveryReceipt}
  alias Sup.Room.RoomService
  alias Sup.Repo
  alias Sup.ScyllaDB
  import Ecto.Query

  def send_message(sender_id, %{"room_id" => room_id, "content" => content, "type" => type}) do
    # Verify user can send to this room
    case RoomService.can_send_message?(sender_id, room_id) do
      true ->
        message_id = generate_message_id()
        timestamp = DateTime.utc_now()

        message_attrs = %{
          id: message_id,
          sender_id: sender_id,
          room_id: room_id,
          content: content,
          type: type,
          timestamp: timestamp
        }

        # Save to ScyllaDB for fast retrieval
        case ScyllaDB.insert_message(message_attrs) do
          :ok ->
            # Create delivery receipts for all room members
            create_delivery_receipts(message_id, room_id, sender_id)

            # Broadcast to room
            broadcast_message(message_attrs)

            {:ok, message_attrs}

          {:error, reason} ->
            {:error, reason}
        end

      false ->
        {:error, "unauthorized"}
    end
  end

  def mark_message_read(message_id, user_id) do
    case Repo.get_by(DeliveryReceipt, message_id: message_id, user_id: user_id) do
      nil ->
        {:error, "receipt_not_found"}

      receipt ->
        changeset =
          DeliveryReceipt.changeset(receipt, %{
            read_at: DateTime.utc_now(),
            status: :read
          })

        case Repo.update(changeset) do
          {:ok, updated_receipt} ->
            # Broadcast read receipt to sender
            broadcast_delivery_receipt(updated_receipt)
            {:ok, updated_receipt}

          {:error, changeset} ->
            {:error, changeset}
        end
    end
  end

  def get_room_messages(room_id, limit \\ 50, before_timestamp \\ nil) do
    ScyllaDB.get_room_messages(room_id, limit, before_timestamp)
  end

  def search_messages(user_id, query, limit \\ 20) do
    # Get user's accessible rooms
    rooms = RoomService.get_user_rooms(user_id)
    room_ids = Enum.map(rooms, & &1.id)

    # Search in ScyllaDB
    ScyllaDB.search_messages(room_ids, query, limit)
  end

  # Private functions
  defp create_delivery_receipts(message_id, room_id, sender_id) do
    # Get all room members except sender
    members =
      RoomService.get_room_members(room_id)
      |> Enum.reject(&(&1.id == sender_id))

    receipts =
      Enum.map(members, fn member ->
        %{
          id: Ecto.UUID.generate(),
          message_id: message_id,
          user_id: member.id,
          status: :sent,
          sent_at: DateTime.utc_now()
        }
      end)

    Repo.insert_all(DeliveryReceipt, receipts)
  end

  defp broadcast_message(message) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{message.room_id}", {:message, message})
  end

  defp broadcast_delivery_receipt(receipt) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{receipt.user_id}", {:delivery_receipt, receipt})
  end

  defp generate_message_id do
    Ecto.UUID.generate()
  end
end
