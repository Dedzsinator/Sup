defmodule Sup.Security.AuditLog do
  @moduledoc """
  Comprehensive audit logging system for security events and user actions.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query
  alias Sup.Repo

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "audit_logs" do
    field(:event_type, :string)
    field(:user_id, :binary_id)
    field(:ip_address, :string)
    field(:user_agent, :string)
    field(:resource_type, :string)
    field(:resource_id, :string)
    field(:action, :string)
    field(:changes, :map)
    field(:metadata, :map)
    field(:severity, :string)
    field(:success, :boolean)
    field(:error_message, :string)
    field(:session_id, :string)
    field(:request_id, :string)

    timestamps(type: :utc_datetime)
  end

  @event_types [
    "authentication",
    "authorization",
    "data_access",
    "data_modification",
    "security_violation",
    "system_event",
    "user_action",
    "admin_action"
  ]

  @severity_levels ["low", "medium", "high", "critical"]

  def changeset(audit_log, attrs) do
    audit_log
    |> cast(attrs, [
      :event_type,
      :user_id,
      :ip_address,
      :user_agent,
      :resource_type,
      :resource_id,
      :action,
      :changes,
      :metadata,
      :severity,
      :success,
      :error_message,
      :session_id,
      :request_id
    ])
    |> validate_required([:event_type, :action, :severity, :success])
    |> validate_inclusion(:event_type, @event_types)
    |> validate_inclusion(:severity, @severity_levels)
  end

  @doc """
  Log an authentication event
  """
  def log_auth_event(type, user_id, conn_or_metadata, success \\ true, error \\ nil) do
    metadata = extract_metadata(conn_or_metadata)

    attrs = %{
      event_type: "authentication",
      user_id: user_id,
      ip_address: metadata[:ip_address],
      user_agent: metadata[:user_agent],
      action: to_string(type),
      success: success,
      error_message: error,
      severity: if(success, do: "low", else: "medium"),
      metadata: metadata,
      session_id: metadata[:session_id],
      request_id: metadata[:request_id]
    }

    create_log(attrs)
  end

  @doc """
  Log a security violation
  """
  def log_security_violation(violation_type, details, conn_or_metadata, severity \\ "high") do
    metadata = extract_metadata(conn_or_metadata)

    attrs = %{
      event_type: "security_violation",
      user_id: metadata[:user_id],
      ip_address: metadata[:ip_address],
      user_agent: metadata[:user_agent],
      action: to_string(violation_type),
      success: false,
      severity: severity,
      metadata: Map.merge(metadata, %{violation_details: details}),
      session_id: metadata[:session_id],
      request_id: metadata[:request_id]
    }

    create_log(attrs)
  end

  @doc """
  Log data access event
  """
  def log_data_access(resource_type, resource_id, action, user_id, conn_or_metadata) do
    metadata = extract_metadata(conn_or_metadata)

    attrs = %{
      event_type: "data_access",
      user_id: user_id,
      resource_type: to_string(resource_type),
      resource_id: to_string(resource_id),
      action: to_string(action),
      ip_address: metadata[:ip_address],
      user_agent: metadata[:user_agent],
      success: true,
      severity: "low",
      metadata: metadata,
      session_id: metadata[:session_id],
      request_id: metadata[:request_id]
    }

    create_log(attrs)
  end

  @doc """
  Log data modification event
  """
  def log_data_modification(
        resource_type,
        resource_id,
        action,
        changes,
        user_id,
        conn_or_metadata
      ) do
    metadata = extract_metadata(conn_or_metadata)

    attrs = %{
      event_type: "data_modification",
      user_id: user_id,
      resource_type: to_string(resource_type),
      resource_id: to_string(resource_id),
      action: to_string(action),
      changes: sanitize_changes(changes),
      ip_address: metadata[:ip_address],
      user_agent: metadata[:user_agent],
      success: true,
      severity: "medium",
      metadata: metadata,
      session_id: metadata[:session_id],
      request_id: metadata[:request_id]
    }

    create_log(attrs)
  end

  @doc """
  Log admin action
  """
  def log_admin_action(action, target_user_id, changes, admin_user_id, conn_or_metadata) do
    metadata = extract_metadata(conn_or_metadata)

    attrs = %{
      event_type: "admin_action",
      user_id: admin_user_id,
      resource_type: "user",
      resource_id: to_string(target_user_id),
      action: to_string(action),
      changes: sanitize_changes(changes),
      ip_address: metadata[:ip_address],
      user_agent: metadata[:user_agent],
      success: true,
      severity: "high",
      metadata: metadata,
      session_id: metadata[:session_id],
      request_id: metadata[:request_id]
    }

    create_log(attrs)
  end

  @doc """
  Get audit logs with filtering and pagination
  """
  def get_logs(filters \\ %{}, opts \\ []) do
    limit = Keyword.get(opts, :limit, 50)
    offset = Keyword.get(opts, :offset, 0)

    query = from(log in __MODULE__)

    query
    |> apply_filters(filters)
    |> order_by([log], desc: log.inserted_at)
    |> limit(^limit)
    |> offset(^offset)
    |> Repo.all()
  end

  @doc """
  Get audit logs count with filtering
  """
  def count_logs(filters \\ %{}) do
    query = from(log in __MODULE__)

    query
    |> apply_filters(filters)
    |> Repo.aggregate(:count, :id)
  end

  @doc """
  Get recent security violations
  """
  def get_recent_violations(hours \\ 24) do
    since = DateTime.utc_now() |> DateTime.add(-hours * 3600, :second)

    from(log in __MODULE__,
      where: log.event_type == "security_violation" and log.inserted_at >= ^since,
      order_by: [desc: log.inserted_at]
    )
    |> Repo.all()
  end

  @doc """
  Get failed login attempts for an IP address
  """
  def get_failed_logins(ip_address, hours \\ 1) do
    since = DateTime.utc_now() |> DateTime.add(-hours * 3600, :second)

    from(log in __MODULE__,
      where:
        log.event_type == "authentication" and
          log.action == "login" and
          log.success == false and
          log.ip_address == ^ip_address and
          log.inserted_at >= ^since
    )
    |> Repo.all()
  end

  @doc """
  Clean up old audit logs (for data retention)
  """
  def cleanup_old_logs(days \\ 90) do
    cutoff_date = DateTime.utc_now() |> DateTime.add(-days * 24 * 3600, :second)

    from(log in __MODULE__, where: log.inserted_at < ^cutoff_date)
    |> Repo.delete_all()
  end

  # Private functions

  defp create_log(attrs) do
    changeset = changeset(%__MODULE__{}, attrs)

    case Repo.insert(changeset) do
      {:ok, log} ->
        # Optionally trigger real-time alerts for critical events
        if attrs.severity == "critical" do
          trigger_security_alert(log)
        end

        {:ok, log}

      {:error, changeset} ->
        {:error, changeset}
    end
  end

  defp extract_metadata(%Plug.Conn{} = conn) do
    %{
      ip_address: get_client_ip(conn),
      user_agent: get_user_agent(conn),
      user_id: get_user_id(conn),
      session_id: get_session_id(conn),
      request_id: get_request_id(conn)
    }
  end

  defp extract_metadata(metadata) when is_map(metadata) do
    metadata
  end

  defp extract_metadata(_), do: %{}

  defp get_client_ip(conn) do
    case Plug.Conn.get_req_header(conn, "x-forwarded-for") do
      [forwarded_ip | _] ->
        forwarded_ip |> String.split(",") |> List.first() |> String.trim()

      [] ->
        case Plug.Conn.get_req_header(conn, "x-real-ip") do
          [real_ip | _] -> real_ip
          [] -> conn.remote_ip |> :inet.ntoa() |> to_string()
        end
    end
  end

  defp get_user_agent(conn) do
    case Plug.Conn.get_req_header(conn, "user-agent") do
      [user_agent | _] -> user_agent
      [] -> nil
    end
  end

  defp get_user_id(conn) do
    case Guardian.Plug.current_resource(conn) do
      %{id: user_id} -> user_id
      _ -> nil
    end
  end

  defp get_session_id(conn) do
    case Plug.Conn.get_session(conn, :session_id) do
      nil -> generate_session_id()
      session_id -> session_id
    end
  end

  defp get_request_id(conn) do
    case Plug.Conn.get_req_header(conn, "x-request-id") do
      [request_id | _] -> request_id
      [] -> generate_request_id()
    end
  end

  defp generate_session_id do
    :crypto.strong_rand_bytes(16) |> Base.encode64() |> binary_part(0, 22)
  end

  defp generate_request_id do
    :crypto.strong_rand_bytes(12) |> Base.encode64() |> binary_part(0, 16)
  end

  defp sanitize_changes(changes) when is_map(changes) do
    # Remove sensitive fields from change logs
    sensitive_fields = [
      :password,
      :password_hash,
      :two_factor_secret,
      :backup_codes,
      :private_key
    ]

    Enum.reduce(sensitive_fields, changes, fn field, acc ->
      if Map.has_key?(acc, field) do
        Map.put(acc, field, "[REDACTED]")
      else
        acc
      end
    end)
  end

  defp sanitize_changes(changes), do: changes

  defp apply_filters(query, filters) do
    Enum.reduce(filters, query, fn
      {:user_id, user_id}, query ->
        where(query, [log], log.user_id == ^user_id)

      {:event_type, event_type}, query ->
        where(query, [log], log.event_type == ^event_type)

      {:action, action}, query ->
        where(query, [log], log.action == ^action)

      {:severity, severity}, query ->
        where(query, [log], log.severity == ^severity)

      {:success, success}, query ->
        where(query, [log], log.success == ^success)

      {:ip_address, ip_address}, query ->
        where(query, [log], log.ip_address == ^ip_address)

      {:from_date, from_date}, query ->
        where(query, [log], log.inserted_at >= ^from_date)

      {:to_date, to_date}, query ->
        where(query, [log], log.inserted_at <= ^to_date)

      _, query ->
        query
    end)
  end

  defp trigger_security_alert(log) do
    # This could integrate with external alerting systems
    # For now, we'll just log to the console
    require Logger
    Logger.error("SECURITY ALERT: #{log.event_type} - #{log.action} from #{log.ip_address}")
  end
end
