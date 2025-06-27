defmodule Sup.Voice.CallService do
  @moduledoc """
  Service for managing voice and video calls.
  """

  alias Sup.Voice.Call
  alias Sup.Repo
  import Ecto.Query

  # Call initiation functions
  def initiate_call(caller_id, opts \\ [])

  def initiate_call(caller_id, opts) when is_list(opts) do
    room_id = Keyword.get(opts, :room_id)
    call_type = Keyword.get(opts, :type, :voice)
    participants = Keyword.get(opts, :participants, [])

    # Add caller to participants
    all_participants = [caller_id | participants] |> Enum.uniq()

    call_attrs = %{
      caller_id: caller_id,
      room_id: room_id,
      type: call_type,
      status: :connecting,
      started_at: DateTime.utc_now(),
      participants: all_participants
    }

    case Call.changeset(%Call{}, call_attrs) |> Repo.insert() do
      {:ok, call} ->
        # Broadcast call initiation to participants
        broadcast_call_event(call, :call_initiated)
        {:ok, Call.public_fields(call)}

      {:error, changeset} ->
        {:error, changeset}
    end
  end

  # Additional overload for initiate_call/2 and initiate_call/3
  def initiate_call(caller_id, room_id) when is_binary(room_id) do
    initiate_call(caller_id, room_id: room_id)
  end

  def initiate_call(caller_id, room_id, call_type) when is_binary(room_id) do
    initiate_call(caller_id, room_id: room_id, type: call_type)
  end

  def answer_call(call_id, user_id) do
    case get_call(call_id) do
      nil ->
        {:error, "call_not_found"}

      %Call{status: :ended} ->
        {:error, "call_ended"}

      %Call{status: :declined} ->
        {:error, "call_declined"}

      call ->
        if user_id in call.participants do
          updated_call =
            call
            |> Call.changeset(%{status: :active})
            |> Repo.update!()

          broadcast_call_event(updated_call, :call_answered)
          {:ok, Call.public_fields(updated_call)}
        else
          {:error, "not_participant"}
        end
    end
  end

  # Alias for answer_call for consistency
  def accept_call(call_id, user_id) do
    answer_call(call_id, user_id)
  end

  # Alias for decline_call
  def reject_call(call_id, user_id) do
    decline_call(call_id, user_id)
  end

  def decline_call(call_id, user_id) do
    case get_call(call_id) do
      nil ->
        {:error, "call_not_found"}

      call ->
        if user_id in call.participants do
          updated_call =
            call
            |> Call.changeset(%{
              status: :declined,
              ended_at: DateTime.utc_now()
            })
            |> Repo.update!()

          broadcast_call_event(updated_call, :call_declined)
          {:ok, Call.public_fields(updated_call)}
        else
          {:error, "not_participant"}
        end
    end
  end

  def end_call(call_id, user_id) do
    case get_call(call_id) do
      nil ->
        {:error, "call_not_found"}

      call ->
        if user_id in call.participants do
          duration =
            if call.started_at do
              DateTime.diff(DateTime.utc_now(), call.started_at, :second)
            else
              0
            end

          updated_call =
            call
            |> Call.changeset(%{
              status: :ended,
              ended_at: DateTime.utc_now(),
              duration: duration
            })
            |> Repo.update!()

          broadcast_call_event(updated_call, :call_ended)
          {:ok, Call.public_fields(updated_call)}
        else
          {:error, "not_participant"}
        end
    end
  end

  def join_call(call_id, user_id) do
    case get_call(call_id) do
      nil ->
        {:error, "call_not_found"}

      %Call{status: status} when status in [:ended, :declined] ->
        {:error, "call_not_active"}

      call ->
        if user_id not in call.participants do
          updated_participants = [user_id | call.participants]

          updated_call =
            call
            |> Call.changeset(%{participants: updated_participants})
            |> Repo.update!()

          broadcast_call_event(updated_call, :participant_joined, user_id)
          {:ok, Call.public_fields(updated_call)}
        else
          {:ok, Call.public_fields(call)}
        end
    end
  end

  def leave_call(call_id, user_id) do
    case get_call(call_id) do
      nil ->
        {:error, "call_not_found"}

      call ->
        if user_id in call.participants do
          updated_participants = List.delete(call.participants, user_id)

          # If no participants left or caller left, end the call
          should_end_call = Enum.empty?(updated_participants) or user_id == call.caller_id

          attrs =
            if should_end_call do
              duration =
                if call.started_at do
                  DateTime.diff(DateTime.utc_now(), call.started_at, :second)
                else
                  0
                end

              %{
                participants: updated_participants,
                status: :ended,
                ended_at: DateTime.utc_now(),
                duration: duration
              }
            else
              %{participants: updated_participants}
            end

          updated_call =
            call
            |> Call.changeset(attrs)
            |> Repo.update!()

          event = if should_end_call, do: :call_ended, else: :participant_left
          broadcast_call_event(updated_call, event, user_id)
          {:ok, Call.public_fields(updated_call)}
        else
          {:error, "not_participant"}
        end
    end
  end

  def get_call(call_id) do
    Repo.get(Call, call_id)
  end

  def get_user_calls(user_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 50)
    status_filter = Keyword.get(opts, :status)

    query =
      from(c in Call,
        where: ^user_id in c.participants,
        order_by: [desc: c.started_at],
        limit: ^limit
      )

    query =
      if status_filter do
        where(query, [c], c.status == ^status_filter)
      else
        query
      end

    query
    |> Repo.all()
    |> Enum.map(&Call.public_fields/1)
  end

  def update_call_quality(call_id, user_id, metrics) do
    case get_call(call_id) do
      nil ->
        {:error, "call_not_found"}

      call ->
        if user_id in call.participants do
          current_metrics = call.quality_metrics || %{}
          updated_metrics = Map.put(current_metrics, user_id, metrics)

          call
          |> Call.changeset(%{quality_metrics: updated_metrics})
          |> Repo.update()
        else
          {:error, "not_participant"}
        end
    end
  end

  # WebRTC signaling helpers
  def create_webrtc_offer(call_id, user_id, offer) do
    broadcast_signaling(call_id, user_id, :offer, offer)
  end

  def create_webrtc_answer(call_id, user_id, answer) do
    broadcast_signaling(call_id, user_id, :answer, answer)
  end

  def add_ice_candidate(call_id, user_id, candidate) do
    broadcast_signaling(call_id, user_id, :ice_candidate, candidate)
  end

  # Handle WebRTC signaling data
  def handle_webrtc_signal(call_id, user_id, signal_data) do
    case signal_data do
      %{"type" => "offer", "sdp" => sdp} ->
        create_webrtc_offer(call_id, user_id, %{sdp: sdp})

      %{"type" => "answer", "sdp" => sdp} ->
        create_webrtc_answer(call_id, user_id, %{sdp: sdp})

      %{"type" => "ice-candidate"} = candidate ->
        add_ice_candidate(call_id, user_id, candidate)

      _ ->
        require Logger
        Logger.warning("Unknown WebRTC signal type: #{inspect(signal_data)}")
        {:error, "unknown_signal_type"}
    end
  end

  # Get active calls for a user
  def get_active_calls(user_id) do
    from(c in Call,
      where: ^user_id in c.participants and c.status in [:connecting, :active],
      order_by: [desc: c.started_at]
    )
    |> Repo.all()
    |> Enum.map(&Call.public_fields/1)
  end

  defp broadcast_call_event(call, event, additional_data \\ nil) do
    # Broadcast to all call participants
    Phoenix.PubSub.broadcast(Sup.PubSub, "calls", %{
      event: event,
      call: Call.public_fields(call),
      data: additional_data
    })

    # Also broadcast to each participant individually
    Enum.each(call.participants, fn participant_id ->
      Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{participant_id}", %{
        type: :call_event,
        event: event,
        call: Call.public_fields(call),
        data: additional_data
      })
    end)
  end

  defp broadcast_signaling(call_id, from_user_id, type, data) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "call:#{call_id}", %{
      type: :webrtc_signaling,
      signaling_type: type,
      from_user_id: from_user_id,
      data: data
    })
  end
end
