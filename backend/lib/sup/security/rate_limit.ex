defmodule Sup.Security.RateLimit do
  @moduledoc """
  Rate limiting functionality using Hammer for the Sup application.
  Provides different rate limits for different types of operations.
  """

  alias Sup.Security.Config

  @doc """
  Check rate limit for API endpoints
  """
  def check_rate_limit(conn_or_socket, limit_type \\ :api) do
    id = get_identifier(conn_or_socket)
    config = get_limit_config(limit_type)

    case Hammer.check_rate("#{limit_type}:#{id}", config.scale, config.limit) do
      {:allow, _count} -> :ok
      {:deny, _limit} -> {:error, :rate_limit_exceeded}
    end
  end

  @doc """
  Check rate limit and return remaining count
  """
  def check_rate_limit_with_info(conn_or_socket, limit_type \\ :api) do
    id = get_identifier(conn_or_socket)
    config = get_limit_config(limit_type)

    case Hammer.check_rate("#{limit_type}:#{id}", config.scale, config.limit) do
      {:allow, count} ->
        remaining = max(0, config.limit - count)

        {:ok,
         %{remaining: remaining, limit: config.limit, reset_time: get_reset_time(config.scale)}}

      {:deny, _limit} ->
        {:error, :rate_limit_exceeded}
    end
  end

  @doc """
  Increment rate limit counter without checking
  """
  def increment_rate_limit(conn_or_socket, limit_type \\ :api) do
    id = get_identifier(conn_or_socket)
    config = get_limit_config(limit_type)

    Hammer.check_rate("#{limit_type}:#{id}", config.scale, config.limit)
  end

  @doc """
  Reset rate limit for a specific identifier
  """
  def reset_rate_limit(identifier, limit_type \\ :api) do
    Hammer.delete_buckets("#{limit_type}:#{identifier}")
  end

  @doc """
  Get current rate limit status without incrementing
  """
  def get_rate_limit_status(conn_or_socket, limit_type \\ :api) do
    id = get_identifier(conn_or_socket)
    config = get_limit_config(limit_type)

    case Hammer.inspect_bucket("#{limit_type}:#{id}", config.scale, config.limit) do
      {:ok, {count, _count_remaining, _ms_to_next_bucket, _created_at, _updated_at}} ->
        remaining = max(0, config.limit - count)

        {:ok,
         %{
           current_count: count,
           remaining: remaining,
           limit: config.limit,
           reset_time: get_reset_time(config.scale)
         }}

      {:error, _reason} ->
        {:ok,
         %{
           current_count: 0,
           remaining: config.limit,
           limit: config.limit,
           reset_time: get_reset_time(config.scale)
         }}
    end
  end

  # Private functions

  defp get_identifier(%Plug.Conn{} = conn) do
    # Try to get user ID first, fallback to IP
    case Guardian.Plug.current_resource(conn) do
      %{id: user_id} -> "user:#{user_id}"
      _ -> "ip:#{get_client_ip(conn)}"
    end
  end

  defp get_identifier(%{assigns: %{user_id: user_id}}) when not is_nil(user_id) do
    "user:#{user_id}"
  end

  defp get_identifier(%{transport_pid: pid}) do
    # For WebSocket connections, use the transport PID
    "ws:#{inspect(pid)}"
  end

  defp get_identifier(socket) when is_map(socket) do
    # Fallback for other socket types
    case Map.get(socket, :remote_ip) do
      ip when not is_nil(ip) -> "ip:#{:inet.ntoa(ip)}"
      _ -> "unknown:#{:erlang.phash2(socket)}"
    end
  end

  defp get_identifier(_), do: "anonymous"

  defp get_client_ip(conn) do
    # Check for forwarded headers first (for reverse proxies)
    case Plug.Conn.get_req_header(conn, "x-forwarded-for") do
      [forwarded_ip | _] ->
        forwarded_ip
        |> String.split(",")
        |> List.first()
        |> String.trim()

      [] ->
        case Plug.Conn.get_req_header(conn, "x-real-ip") do
          [real_ip | _] ->
            real_ip

          [] ->
            conn.remote_ip
            |> :inet.ntoa()
            |> to_string()
        end
    end
  end

  defp get_limit_config(limit_type) do
    Config.rate_limit_config()
    |> Map.get(limit_type, Config.rate_limit_config().api)
  end

  defp get_reset_time(scale_ms) do
    DateTime.utc_now()
    |> DateTime.add(scale_ms, :millisecond)
    |> DateTime.to_unix()
  end
end
