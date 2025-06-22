defmodule Sup.Security.HeadersPlug do
  @moduledoc """
  Plug for adding security headers to HTTP responses.
  """

  import Plug.Conn
  alias Sup.Security.Config

  def init(opts), do: opts

  def call(conn, _opts) do
    headers = Config.security_headers()

    Enum.reduce(headers, conn, fn {header, value}, acc_conn ->
      put_resp_header(acc_conn, header, value)
    end)
  end
end
