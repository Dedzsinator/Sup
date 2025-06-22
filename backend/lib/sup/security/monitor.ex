defmodule Sup.Security.Monitor do
  @moduledoc """
  Real-time security monitoring and intrusion detection system.
  Monitors for suspicious activities, failed login attempts, rate limit violations, etc.
  """

  use GenServer
  require Logger
  alias Sup.Security.{AuditLog, RateLimit}
  alias Sup.Repo
  import Ecto.Query

  # Check every minute
  @check_interval :timer.minutes(1)
  @alert_thresholds %{
    # 10 failed logins from same IP in 1 hour
    failed_logins_per_ip: 10,
    # 5 failed logins for same user in 1 hour
    failed_logins_per_user: 5,
    # 50 rate limit hits in 5 minutes
    rate_limit_violations: 50,
    # 5 requests with suspicious user agents
    suspicious_user_agents: 5,
    # 3 logins from different countries in 1 hour
    geo_anomalies: 3,
    # 10 admin actions in 5 minutes
    admin_actions_burst: 10
  }

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  # Private function to get suspicious patterns
  defp suspicious_patterns do
    [
      ~r/bot|crawler|spider/i,
      ~r/scan|probe|exploit/i,
      ~r/injection|xss|csrf/i,
      ~r/burp|nmap|sqlmap/i
    ]
  end

  @doc """
  Report a security event for monitoring
  """
  def report_event(event_type, details, metadata \\ %{}) do
    GenServer.cast(__MODULE__, {:security_event, event_type, details, metadata})
  end

  @doc """
  Get current security alerts
  """
  def get_alerts do
    GenServer.call(__MODULE__, :get_alerts)
  end

  @doc """
  Get security statistics
  """
  def get_security_stats do
    GenServer.call(__MODULE__, :get_security_stats)
  end

  @doc """
  Check if IP address is suspicious
  """
  def is_suspicious_ip?(ip_address) do
    GenServer.call(__MODULE__, {:check_suspicious_ip, ip_address})
  end

  # GenServer Callbacks

  @impl true
  def init(_opts) do
    schedule_monitoring()

    state = %{
      alerts: [],
      suspicious_ips: MapSet.new(),
      blocked_ips: MapSet.new(),
      security_stats: init_stats()
    }

    {:ok, state}
  end

  @impl true
  def handle_cast({:security_event, event_type, details, metadata}, state) do
    updated_state = process_security_event(event_type, details, metadata, state)
    {:noreply, updated_state}
  end

  @impl true
  def handle_call(:get_alerts, _from, state) do
    {:reply, state.alerts, state}
  end

  @impl true
  def handle_call(:get_security_stats, _from, state) do
    {:reply, state.security_stats, state}
  end

  @impl true
  def handle_call({:check_suspicious_ip, ip_address}, _from, state) do
    is_suspicious =
      MapSet.member?(state.suspicious_ips, ip_address) or
        MapSet.member?(state.blocked_ips, ip_address)

    {:reply, is_suspicious, state}
  end

  @impl true
  def handle_info(:monitor, state) do
    updated_state = perform_monitoring_checks(state)
    schedule_monitoring()
    {:noreply, updated_state}
  end

  # Private Functions

  defp schedule_monitoring do
    Process.send_after(self(), :monitor, @check_interval)
  end

  defp init_stats do
    %{
      total_events: 0,
      failed_logins_today: 0,
      rate_limit_violations_today: 0,
      suspicious_activities_today: 0,
      blocked_ips_count: 0,
      active_alerts: 0,
      last_updated: DateTime.utc_now()
    }
  end

  defp process_security_event(event_type, details, metadata, state) do
    # Update statistics
    updated_stats = update_security_stats(state.security_stats, event_type)

    # Check for immediate threats
    {alerts, suspicious_ips, blocked_ips} =
      analyze_security_event(event_type, details, metadata, state)

    %{
      state
      | security_stats: updated_stats,
        alerts: alerts ++ state.alerts,
        suspicious_ips: MapSet.union(state.suspicious_ips, suspicious_ips),
        blocked_ips: MapSet.union(state.blocked_ips, blocked_ips)
    }
  end

  defp update_security_stats(stats, event_type) do
    base_update = %{
      stats
      | total_events: stats.total_events + 1,
        last_updated: DateTime.utc_now()
    }

    case event_type do
      :failed_login ->
        %{base_update | failed_logins_today: stats.failed_logins_today + 1}

      :rate_limit_violation ->
        %{base_update | rate_limit_violations_today: stats.rate_limit_violations_today + 1}

      :suspicious_activity ->
        %{base_update | suspicious_activities_today: stats.suspicious_activities_today + 1}

      _ ->
        base_update
    end
  end

  defp analyze_security_event(event_type, details, metadata, state) do
    case event_type do
      :failed_login ->
        analyze_failed_login(details, metadata, state)

      :rate_limit_violation ->
        analyze_rate_limit_violation(details, metadata, state)

      :suspicious_user_agent ->
        analyze_suspicious_user_agent(details, metadata, state)

      :geo_anomaly ->
        analyze_geo_anomaly(details, metadata, state)

      _ ->
        {[], MapSet.new(), MapSet.new()}
    end
  end

  defp analyze_failed_login(%{ip_address: ip, user_id: user_id}, metadata, state) do
    alerts = []
    suspicious_ips = MapSet.new()
    blocked_ips = MapSet.new()

    # Check failed logins from same IP
    recent_failures_ip = get_recent_failed_logins_by_ip(ip, 1)

    {alerts, suspicious_ips, blocked_ips} =
      if length(recent_failures_ip) >= @alert_thresholds.failed_logins_per_ip do
        alert =
          create_alert(:brute_force_ip, %{
            ip_address: ip,
            failed_attempts: length(recent_failures_ip),
            severity: "high"
          })

        # Block IP temporarily
        AuditLog.log_security_violation(
          "ip_blocked_brute_force",
          %{ip_address: ip, failed_attempts: length(recent_failures_ip)},
          metadata,
          "high"
        )

        {[alert | alerts], MapSet.put(suspicious_ips, ip), MapSet.put(blocked_ips, ip)}
      else
        {alerts, suspicious_ips, blocked_ips}
      end

    # Check failed logins for same user
    if user_id do
      recent_failures_user = get_recent_failed_logins_by_user(user_id, 1)

      if length(recent_failures_user) >= @alert_thresholds.failed_logins_per_user do
        alert =
          create_alert(:brute_force_user, %{
            user_id: user_id,
            failed_attempts: length(recent_failures_user),
            severity: "medium"
          })

        {[alert | alerts], suspicious_ips, blocked_ips}
      else
        {alerts, suspicious_ips, blocked_ips}
      end
    else
      {alerts, suspicious_ips, blocked_ips}
    end
  end

  defp analyze_rate_limit_violation(%{ip_address: ip, endpoint: endpoint}, metadata, _state) do
    # 5 minutes
    recent_violations = get_recent_rate_limit_violations(ip, 5)

    if length(recent_violations) >= @alert_thresholds.rate_limit_violations do
      alert =
        create_alert(:rate_limit_abuse, %{
          ip_address: ip,
          endpoint: endpoint,
          violations: length(recent_violations),
          severity: "medium"
        })

      AuditLog.log_security_violation(
        "rate_limit_abuse",
        %{ip_address: ip, endpoint: endpoint, violations: length(recent_violations)},
        metadata,
        "medium"
      )

      {[alert], MapSet.new([ip]), MapSet.new()}
    else
      {[], MapSet.new(), MapSet.new()}
    end
  end

  defp analyze_suspicious_user_agent(%{user_agent: user_agent, ip_address: ip}, metadata, _state) do
    if is_suspicious_user_agent?(user_agent) do
      alert =
        create_alert(:suspicious_user_agent, %{
          ip_address: ip,
          user_agent: user_agent,
          severity: "low"
        })

      AuditLog.log_security_violation(
        "suspicious_user_agent",
        %{ip_address: ip, user_agent: user_agent},
        metadata,
        "low"
      )

      {[alert], MapSet.new([ip]), MapSet.new()}
    else
      {[], MapSet.new(), MapSet.new()}
    end
  end

  defp analyze_geo_anomaly(
         %{user_id: user_id, country: country, ip_address: ip},
         metadata,
         _state
       ) do
    # 1 hour
    recent_countries = get_recent_login_countries(user_id, 1)

    if length(Enum.uniq(recent_countries)) >= @alert_thresholds.geo_anomalies do
      alert =
        create_alert(:geo_anomaly, %{
          user_id: user_id,
          countries: recent_countries,
          current_country: country,
          ip_address: ip,
          severity: "high"
        })

      AuditLog.log_security_violation(
        "geographical_anomaly",
        %{user_id: user_id, countries: recent_countries, current_country: country},
        metadata,
        "high"
      )

      {[alert], MapSet.new(), MapSet.new()}
    else
      {[], MapSet.new(), MapSet.new()}
    end
  end

  defp perform_monitoring_checks(state) do
    # Perform periodic security checks
    state
    |> check_audit_log_anomalies()
    |> check_system_health()
    |> cleanup_old_alerts()
  end

  defp check_audit_log_anomalies(state) do
    # Check for unusual patterns in audit logs
    # Last hour
    recent_violations = AuditLog.get_recent_violations(1)

    alerts =
      recent_violations
      |> Enum.group_by(& &1.ip_address)
      |> Enum.filter(fn {_ip, violations} -> length(violations) > 5 end)
      |> Enum.map(fn {ip, violations} ->
        create_alert(:repeated_violations, %{
          ip_address: ip,
          violation_count: length(violations),
          severity: "medium"
        })
      end)

    %{state | alerts: alerts ++ state.alerts}
  end

  defp check_system_health(state) do
    # Check system metrics and resource usage
    # This would integrate with telemetry and system monitoring

    # For now, just log current stats
    Logger.info("Security Monitor Stats: #{inspect(state.security_stats)}")
    state
  end

  defp cleanup_old_alerts(state) do
    # Remove alerts older than 24 hours
    cutoff_time = DateTime.utc_now() |> DateTime.add(-24 * 3600, :second)

    fresh_alerts =
      Enum.filter(state.alerts, fn alert ->
        DateTime.compare(alert.created_at, cutoff_time) == :gt
      end)

    updated_stats = %{state.security_stats | active_alerts: length(fresh_alerts)}

    %{state | alerts: fresh_alerts, security_stats: updated_stats}
  end

  defp create_alert(type, details) do
    %{
      id: :crypto.strong_rand_bytes(8) |> Base.encode64() |> binary_part(0, 11),
      type: type,
      details: details,
      created_at: DateTime.utc_now(),
      acknowledged: false
    }
  end

  defp get_recent_failed_logins_by_ip(ip_address, hours) do
    since = DateTime.utc_now() |> DateTime.add(-hours * 3600, :second)

    AuditLog.get_logs(%{
      event_type: "authentication",
      action: "login",
      success: false,
      ip_address: ip_address,
      from_date: since
    })
  end

  defp get_recent_failed_logins_by_user(user_id, hours) do
    since = DateTime.utc_now() |> DateTime.add(-hours * 3600, :second)

    AuditLog.get_logs(%{
      event_type: "authentication",
      action: "login",
      success: false,
      user_id: user_id,
      from_date: since
    })
  end

  defp get_recent_rate_limit_violations(ip_address, minutes) do
    since = DateTime.utc_now() |> DateTime.add(-minutes * 60, :second)

    AuditLog.get_logs(%{
      event_type: "security_violation",
      action: "rate_limit_exceeded",
      ip_address: ip_address,
      from_date: since
    })
  end

  defp get_recent_login_countries(user_id, hours) do
    since = DateTime.utc_now() |> DateTime.add(-hours * 3600, :second)

    logs =
      AuditLog.get_logs(%{
        event_type: "authentication",
        action: "login",
        success: true,
        user_id: user_id,
        from_date: since
      })

    Enum.map(logs, fn log ->
      get_in(log.metadata, ["country"]) || "unknown"
    end)
  end

  defp is_suspicious_user_agent?(user_agent) when is_binary(user_agent) do
    Enum.any?(suspicious_patterns(), fn pattern ->
      Regex.match?(pattern, user_agent)
    end)
  end

  defp is_suspicious_user_agent?(_), do: false
end
