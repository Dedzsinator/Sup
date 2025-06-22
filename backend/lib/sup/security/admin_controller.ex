defmodule Sup.Security.AdminController do
  @moduledoc """
  Admin API endpoints for security monitoring and management.
  Requires admin role for access.
  """

  import Plug.Conn
  import Ecto.Query
  require Logger

  alias Sup.Security.{AuditLog, Monitor, RBAC, RateLimit}
  alias Sup.Auth.User
  alias Sup.Repo

  @doc """
  Get security dashboard overview
  """
  def security_dashboard(conn, _params) do
    user = Guardian.Plug.current_resource(conn)

    unless RBAC.has_permission?(user, "admin", "read") do
      send_resp(conn, 403, Jason.encode!(%{error: "Insufficient permissions"}))
    else
      dashboard_data = %{
        security_stats: Monitor.get_security_stats(),
        active_alerts: Monitor.get_alerts(),
        # Last 24 hours
        recent_violations: AuditLog.get_recent_violations(24),
        audit_summary: get_audit_summary(),
        user_security_summary: get_user_security_summary(),
        system_health: get_system_health_summary()
      }

      # Log admin access
      AuditLog.log_admin_action(
        "security_dashboard_access",
        nil,
        %{},
        user.id,
        conn
      )

      send_resp(conn, 200, Jason.encode!(dashboard_data))
    end
  end

  @doc """
  Get detailed audit logs with filtering
  """
  def audit_logs(conn, params) do
    user = Guardian.Plug.current_resource(conn)

    unless RBAC.has_permission?(user, "audit", "read") do
      send_resp(conn, 403, Jason.encode!(%{error: "Insufficient permissions"}))
    else
      filters = extract_audit_filters(params)
      limit = String.to_integer(params["limit"] || "50")
      offset = String.to_integer(params["offset"] || "0")

      logs = AuditLog.get_logs(filters, limit: limit, offset: offset)
      total_count = AuditLog.count_logs(filters)

      response = %{
        logs: logs,
        pagination: %{
          limit: limit,
          offset: offset,
          total: total_count,
          has_more: offset + limit < total_count
        }
      }

      send_resp(conn, 200, Jason.encode!(response))
    end
  end

  @doc """
  Export audit logs to CSV
  """
  def export_audit_logs(conn, params) do
    user = Guardian.Plug.current_resource(conn)

    unless RBAC.has_permission?(user, "audit", "export") do
      send_resp(conn, 403, Jason.encode!(%{error: "Insufficient permissions"}))
    else
      filters = extract_audit_filters(params)
      # Max 10k records for export
      logs = AuditLog.get_logs(filters, limit: 10000)

      csv_content = generate_audit_csv(logs)
      filename = "audit_logs_#{Date.utc_today()}.csv"

      # Log export action
      AuditLog.log_admin_action(
        "audit_export",
        nil,
        %{records_count: length(logs), filters: filters},
        user.id,
        conn
      )

      conn
      |> put_resp_header("content-type", "text/csv")
      |> put_resp_header("content-disposition", "attachment; filename=\"#{filename}\"")
      |> send_resp(200, csv_content)
    end
  end

  @doc """
  Get security alerts
  """
  def security_alerts(conn, _params) do
    user = Guardian.Plug.current_resource(conn)

    unless RBAC.has_permission?(user, "system", "read") do
      send_resp(conn, 403, Jason.encode!(%{error: "Insufficient permissions"}))
    else
      alerts = Monitor.get_alerts()
      send_resp(conn, 200, Jason.encode!(%{alerts: alerts}))
    end
  end

  @doc """
  Acknowledge security alert
  """
  def acknowledge_alert(conn, %{"alert_id" => alert_id}) do
    user = Guardian.Plug.current_resource(conn)

    unless RBAC.has_permission?(user, "system", "update") do
      send_resp(conn, 403, Jason.encode!(%{error: "Insufficient permissions"}))
    else
      # In a real implementation, you'd update the alert in the Monitor
      # For now, we'll just log the acknowledgment
      AuditLog.log_admin_action(
        "alert_acknowledged",
        nil,
        %{alert_id: alert_id},
        user.id,
        conn
      )

      send_resp(conn, 200, Jason.encode!(%{status: "acknowledged"}))
    end
  end

  @doc """
  Block/unblock IP address
  """
  def manage_ip_blocking(conn, %{"action" => action, "ip_address" => ip_address}) do
    user = Guardian.Plug.current_resource(conn)

    unless RBAC.has_permission?(user, "system", "update") do
      send_resp(conn, 403, Jason.encode!(%{error: "Insufficient permissions"}))
    else
      case action do
        "block" ->
          # In production, this would integrate with firewall/load balancer
          AuditLog.log_admin_action(
            "ip_blocked",
            nil,
            %{ip_address: ip_address, reason: "manual_admin_action"},
            user.id,
            conn
          )

          send_resp(conn, 200, Jason.encode!(%{status: "blocked", ip_address: ip_address}))

        "unblock" ->
          AuditLog.log_admin_action(
            "ip_unblocked",
            nil,
            %{ip_address: ip_address},
            user.id,
            conn
          )

          send_resp(conn, 200, Jason.encode!(%{status: "unblocked", ip_address: ip_address}))

        _ ->
          send_resp(conn, 400, Jason.encode!(%{error: "Invalid action"}))
      end
    end
  end

  @doc """
  Manage user account security
  """
  def manage_user_security(conn, %{"user_id" => user_id, "action" => action}) do
    admin_user = Guardian.Plug.current_resource(conn)

    unless RBAC.has_permission?(admin_user, "user", "update") do
      send_resp(conn, 403, Jason.encode!(%{error: "Insufficient permissions"}))
    else
      target_user = Repo.get(User, user_id)

      if target_user do
        case action do
          "lock_account" ->
            lock_user_account(target_user, admin_user, conn)

          "unlock_account" ->
            unlock_user_account(target_user, admin_user, conn)

          "reset_failed_attempts" ->
            reset_failed_login_attempts(target_user, admin_user, conn)

          "disable_2fa" ->
            disable_user_2fa(target_user, admin_user, conn)

          _ ->
            send_resp(conn, 400, Jason.encode!(%{error: "Invalid action"}))
        end
      else
        send_resp(conn, 404, Jason.encode!(%{error: "User not found"}))
      end
    end
  end

  @doc """
  Get rate limiting statistics
  """
  def rate_limit_stats(conn, _params) do
    user = Guardian.Plug.current_resource(conn)

    unless RBAC.has_permission?(user, "system", "read") do
      send_resp(conn, 403, Jason.encode!(%{error: "Insufficient permissions"}))
    else
      # This would integrate with your rate limiting system
      # For now, return mock data
      stats = %{
        total_requests_today: 15420,
        rate_limited_requests: 234,
        top_limited_ips: [
          %{ip: "192.168.1.100", hits: 45},
          %{ip: "10.0.0.55", hits: 32},
          %{ip: "172.16.0.20", hits: 28}
        ],
        endpoints: %{
          "/api/auth/login" => %{requests: 1240, limited: 89},
          "/api/messages" => %{requests: 8930, limited: 45},
          "/api/users" => %{requests: 2340, limited: 12}
        }
      }

      send_resp(conn, 200, Jason.encode!(stats))
    end
  end

  # Private helper functions

  defp extract_audit_filters(params) do
    filters = %{}

    filters =
      if params["user_id"], do: Map.put(filters, :user_id, params["user_id"]), else: filters

    filters =
      if params["event_type"],
        do: Map.put(filters, :event_type, params["event_type"]),
        else: filters

    filters = if params["action"], do: Map.put(filters, :action, params["action"]), else: filters

    filters =
      if params["severity"], do: Map.put(filters, :severity, params["severity"]), else: filters

    filters =
      if params["success"],
        do: Map.put(filters, :success, params["success"] == "true"),
        else: filters

    filters =
      if params["ip_address"],
        do: Map.put(filters, :ip_address, params["ip_address"]),
        else: filters

    # Date filters
    filters =
      if params["from_date"] do
        {:ok, date, _} = DateTime.from_iso8601(params["from_date"])
        Map.put(filters, :from_date, date)
      else
        filters
      end

    filters =
      if params["to_date"] do
        {:ok, date, _} = DateTime.from_iso8601(params["to_date"])
        Map.put(filters, :to_date, date)
      else
        filters
      end

    filters
  end

  defp get_audit_summary do
    today = Date.utc_today()
    start_of_day = DateTime.new!(today, ~T[00:00:00])

    %{
      total_events_today: AuditLog.count_logs(%{from_date: start_of_day}),
      failed_logins_today:
        AuditLog.count_logs(%{
          event_type: "authentication",
          success: false,
          from_date: start_of_day
        }),
      security_violations_today:
        AuditLog.count_logs(%{
          event_type: "security_violation",
          from_date: start_of_day
        }),
      admin_actions_today:
        AuditLog.count_logs(%{
          event_type: "admin_action",
          from_date: start_of_day
        })
    }
  end

  defp get_user_security_summary do
    # This would query user statistics
    %{
      total_users: Repo.aggregate(User, :count, :id),
      locked_accounts:
        Repo.aggregate(from(u in User, where: u.account_locked == true), :count, :id),
      users_with_2fa:
        Repo.aggregate(from(u in User, where: u.two_factor_enabled == true), :count, :id),
      admin_users: Repo.aggregate(from(u in User, where: u.role == "admin"), :count, :id)
    }
  end

  defp get_system_health_summary do
    %{
      uptime: System.uptime(),
      memory_usage: :erlang.memory(),
      process_count: length(Process.list()),
      timestamp: DateTime.utc_now()
    }
  end

  defp generate_audit_csv(logs) do
    headers = [
      "ID",
      "Event Type",
      "User ID",
      "IP Address",
      "Action",
      "Success",
      "Severity",
      "Timestamp",
      "Details"
    ]

    rows =
      Enum.map(logs, fn log ->
        [
          log.id,
          log.event_type,
          log.user_id || "",
          log.ip_address || "",
          log.action,
          log.success,
          log.severity,
          DateTime.to_iso8601(log.inserted_at),
          inspect(log.metadata)
        ]
      end)

    [headers | rows]
    |> Enum.map(&Enum.join(&1, ","))
    |> Enum.join("\n")
  end

  defp lock_user_account(user, admin_user, conn) do
    # 24 hours
    locked_until = DateTime.utc_now() |> DateTime.add(24 * 3600, :second)

    changeset =
      User.changeset(user, %{
        account_locked: true,
        locked_until: locked_until
      })

    case Repo.update(changeset) do
      {:ok, _updated_user} ->
        AuditLog.log_admin_action(
          "account_locked",
          user.id,
          %{locked_until: locked_until},
          admin_user.id,
          conn
        )

        send_resp(conn, 200, Jason.encode!(%{status: "locked", locked_until: locked_until}))

      {:error, _changeset} ->
        send_resp(conn, 500, Jason.encode!(%{error: "Failed to lock account"}))
    end
  end

  defp unlock_user_account(user, admin_user, conn) do
    changeset =
      User.changeset(user, %{
        account_locked: false,
        locked_until: nil,
        failed_login_attempts: 0
      })

    case Repo.update(changeset) do
      {:ok, _updated_user} ->
        AuditLog.log_admin_action(
          "account_unlocked",
          user.id,
          %{},
          admin_user.id,
          conn
        )

        send_resp(conn, 200, Jason.encode!(%{status: "unlocked"}))

      {:error, _changeset} ->
        send_resp(conn, 500, Jason.encode!(%{error: "Failed to unlock account"}))
    end
  end

  defp reset_failed_login_attempts(user, admin_user, conn) do
    changeset = User.changeset(user, %{failed_login_attempts: 0})

    case Repo.update(changeset) do
      {:ok, _updated_user} ->
        AuditLog.log_admin_action(
          "failed_attempts_reset",
          user.id,
          %{},
          admin_user.id,
          conn
        )

        send_resp(conn, 200, Jason.encode!(%{status: "reset"}))

      {:error, _changeset} ->
        send_resp(conn, 500, Jason.encode!(%{error: "Failed to reset failed attempts"}))
    end
  end

  defp disable_user_2fa(user, admin_user, conn) do
    changeset =
      User.changeset(user, %{
        two_factor_enabled: false,
        two_factor_secret: nil,
        backup_codes: []
      })

    case Repo.update(changeset) do
      {:ok, _updated_user} ->
        AuditLog.log_admin_action(
          "2fa_disabled",
          user.id,
          %{reason: "admin_action"},
          admin_user.id,
          conn
        )

        send_resp(conn, 200, Jason.encode!(%{status: "2fa_disabled"}))

      {:error, _changeset} ->
        send_resp(conn, 500, Jason.encode!(%{error: "Failed to disable 2FA"}))
    end
  end
end
