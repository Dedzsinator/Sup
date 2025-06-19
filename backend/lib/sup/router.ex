defmodule Sup.Router do
  @moduledoc """
  Main HTTP router using Plug - handles REST API and WebSocket upgrades.
  """

  use Plug.Router
  require Logger

  plug(Plug.Logger)

  plug(CORSPlug,
    origin: ["http://localhost:3000", "http://localhost:19006", "http://localhost:8081"],
    headers: ["authorization", "content-type", "x-requested-with"],
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
  )

  plug(Plug.Parsers, parsers: [:json], json_decoder: Jason)
  plug(:match)
  plug(:dispatch)

  # Health check endpoint
  get "/health" do
    send_resp(conn, 200, Jason.encode!(%{status: "healthy", timestamp: DateTime.utc_now()}))
  end

  # Handle CORS preflight requests
  options "/auth/register" do
    send_resp(conn, 204, "")
  end

  options "/auth/login" do
    send_resp(conn, 204, "")
  end

  # Authentication routes
  post "/auth/register" do
    with {:ok, params} <- validate_register_params(conn.body_params),
         {:ok, user, token} <- Sup.Auth.Service.register(params) do
      send_resp(conn, 201, Jason.encode!(%{user: user, token: token}))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  post "/auth/login" do
    with {:ok, params} <- validate_login_params(conn.body_params),
         {:ok, user, token} <- Sup.Auth.Service.login(params) do
      send_resp(conn, 200, Jason.encode!(%{user: user, token: token}))
    else
      {:error, reason} ->
        send_resp(conn, 401, Jason.encode!(%{error: reason}))
    end
  end

  # WebSocket upgrade for real-time messaging
  get "/ws" do
    case get_req_header(conn, "authorization") do
      ["Bearer " <> token] ->
        case Sup.Auth.Guardian.decode_and_verify(token) do
          {:ok, %{"sub" => user_id}} ->
            conn
            |> WebSockAdapter.upgrade(Sup.WebSocket.Handler, %{user_id: user_id}, timeout: 60_000)
            |> halt()

          _ ->
            send_resp(conn, 401, Jason.encode!(%{error: "invalid_token"}))
        end

      _ ->
        send_resp(conn, 401, Jason.encode!(%{error: "missing_token"}))
    end
  end

  # Public autocomplete endpoints (no auth required for demo)
  post "/autocomplete/suggest" do
    alias Sup.Autocomplete.Service, as: AutocompleteService
    
    with {:ok, params} <- validate_autocomplete_params(conn.body_params),
         {:ok, suggestions} <- AutocompleteService.get_suggestions(
           params["text"],
           [
             user_id: Map.get(params, "user_id"),
             limit: Map.get(params, "max_suggestions", 5)
           ]
         ) do
      send_resp(conn, 200, Jason.encode!(suggestions))
    else
      {:error, reason} ->
        send_resp(conn, 400, Jason.encode!(%{error: reason}))
    end
  end

  get "/autocomplete/health" do
    alias Sup.Autocomplete.Service, as: AutocompleteService
    
    case AutocompleteService.health_check() do
      {:ok, status} ->
        send_resp(conn, 200, Jason.encode!(status))
      {:error, reason} ->
        send_resp(conn, 503, Jason.encode!(%{error: reason}))
    end
  end

  # Protected API routes (require authentication)
  forward("/api", to: Sup.ApiRouter, init_opts: [])

  # Catch-all
  match _ do
    send_resp(conn, 404, Jason.encode!(%{error: "not_found"}))
  end

  # Helper function for autocomplete validation
  defp validate_autocomplete_params(params) when is_map(params) do
    text = Map.get(params, "text")
    
    cond do
      is_nil(text) or text == "" ->
        {:error, "text_required"}
      
      not is_binary(text) ->
        {:error, "text_must_be_string"}
      
      String.length(text) > 500 ->
        {:error, "text_too_long"}
      
      true ->
        {:ok, params}
    end
  end
  
  defp validate_autocomplete_params(_), do: {:error, "invalid_params"}

  # Helper functions
  defp validate_register_params(%{
         "email" => email,
         "password" => password,
         "username" => username
       })
       when is_binary(email) and is_binary(password) and is_binary(username) do
    {:ok, %{email: email, password: password, username: username}}
  end

  defp validate_register_params(_), do: {:error, "invalid_params"}

  defp validate_login_params(%{"email" => email, "password" => password})
       when is_binary(email) and is_binary(password) do
    {:ok, %{email: email, password: password}}
  end

  defp validate_login_params(_), do: {:error, "invalid_params"}
end
