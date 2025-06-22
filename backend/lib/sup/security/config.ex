defmodule Sup.Security.Config do
  @moduledoc """
  Centralized security configuration for the Sup application.
  """

  @doc """
  JWT configuration with enhanced security settings
  """
  def jwt_config do
    %{
      secret_key: Application.get_env(:sup, :secret_key_base),
      issuer: "sup_app",
      ttl: {30, :minutes},
      refresh_ttl: {7, :days},
      algorithm: "HS256",
      allowed_algos: ["HS256"],
      verify_issuer: true
    }
  end

  @doc """
  Rate limiting configuration
  """
  def rate_limit_config do
    %{
      # General API rate limiting
      api: %{
        # 1 minute window
        scale: 60_000,
        # 100 requests per minute
        limit: 100,
        cleanup_rate: 10_000
      },
      # Authentication endpoints - stricter limits
      auth: %{
        # 1 minute window
        scale: 60_000,
        # 5 login attempts per minute
        limit: 5,
        cleanup_rate: 10_000
      },
      # WebSocket connections
      websocket: %{
        # 1 minute window
        scale: 60_000,
        # 50 connection attempts per minute
        limit: 50,
        cleanup_rate: 10_000
      },
      # Message sending
      messages: %{
        # 1 minute window
        scale: 60_000,
        # 60 messages per minute
        limit: 60,
        cleanup_rate: 10_000
      }
    }
  end

  @doc """
  Two-Factor Authentication configuration
  """
  def tfa_config do
    %{
      issuer: "Sup Messaging",
      digits: 6,
      period: 30,
      window: 1,
      backup_codes_count: 10
    }
  end

  @doc """
  Security headers configuration
  """
  def security_headers do
    %{
      "X-Frame-Options" => "DENY",
      "X-Content-Type-Options" => "nosniff",
      "X-XSS-Protection" => "1; mode=block",
      "Strict-Transport-Security" => "max-age=31536000; includeSubDomains",
      "Content-Security-Policy" =>
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' ws: wss:",
      "Referrer-Policy" => "strict-origin-when-cross-origin",
      "Permissions-Policy" => "geolocation=(), microphone=(), camera=()"
    }
  end

  @doc """
  CORS configuration with enhanced security
  """
  def cors_config do
    %{
      origin: allowed_origins(),
      credentials: true,
      max_age: 86400,
      headers: ["Authorization", "Content-Type", "X-Requested-With"],
      methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    }
  end

  @doc """
  WebSocket security configuration
  """
  def websocket_config do
    %{
      allowed_origins: allowed_origins(),
      max_connections_per_ip: 5,
      connection_timeout: 60_000,
      heartbeat_interval: 30_000
    }
  end

  @doc """
  Audit logging configuration
  """
  def audit_config do
    %{
      repo: Sup.Repo,
      table_name: "audit_logs",
      track_all_changes?: false,
      log_only_changes?: true,
      excluded_fields: [:password_hash, :two_factor_secret, :backup_codes]
    }
  end

  @doc """
  Message encryption configuration
  """
  def encryption_config do
    %{
      algorithm: :aes_256_gcm,
      key_derivation: :pbkdf2,
      iterations: 100_000,
      key_length: 32,
      iv_length: 12,
      tag_length: 16
    }
  end

  @doc """
  Session security configuration
  """
  def session_config do
    %{
      # 7 days
      max_age: 3600 * 24 * 7,
      httponly: true,
      secure: Mix.env() == :prod,
      same_site: "strict"
    }
  end

  defp allowed_origins do
    case Mix.env() do
      :prod ->
        System.get_env("ALLOWED_ORIGINS", "https://yourdomain.com")
        |> String.split(",")
        |> Enum.map(&String.trim/1)

      :dev ->
        ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:19006"]

      :test ->
        ["http://localhost:3000"]
    end
  end
end
