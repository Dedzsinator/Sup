defmodule Sup.Security.RateLimitPlug do
  @moduledoc """
  Plug for rate limiting HTTP requests.
  """

  import Plug.Conn
  alias Sup.Security.RateLimit

  def init(opts) do
    Keyword.get(opts, :limit_type, :api)
  end

  def call(conn, limit_type) do
    case RateLimit.check_rate_limit_with_info(conn, limit_type) do
      {:ok, info} ->
        conn
        |> put_resp_header("x-ratelimit-limit", to_string(info.limit))
        |> put_resp_header("x-ratelimit-remaining", to_string(info.remaining))
        |> put_resp_header("x-ratelimit-reset", to_string(info.reset_time))

      {:error, :rate_limit_exceeded} ->
        conn
        |> put_status(429)
        |> put_resp_content_type("application/json")
        |> send_resp(
          429,
          Jason.encode!(%{
            error: "Rate limit exceeded",
            message: "Too many requests. Please try again later.",
            retry_after: 60
          })
        )
        |> halt()
    end
  end
end
