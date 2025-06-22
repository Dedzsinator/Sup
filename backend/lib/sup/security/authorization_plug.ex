defmodule Sup.Security.AuthorizationPlug do
  @moduledoc """
  Authorization plug that enforces RBAC permissions for API endpoints.
  """

  import Plug.Conn
  alias Sup.Security.{RBAC, AuditLog}

  def init(opts), do: opts

  def call(conn, opts) do
    required_permissions = get_required_permissions(conn, opts)

    case Guardian.Plug.current_resource(conn) do
      nil ->
        # No authenticated user
        if Enum.empty?(required_permissions) do
          conn
        else
          unauthorized_response(conn, "Authentication required")
        end

      user ->
        if check_permissions(user, required_permissions, conn) do
          conn
        else
          # Log authorization failure
          AuditLog.log_security_violation(
            "authorization_failure",
            %{
              required_permissions: required_permissions,
              user_role: user.role,
              endpoint: "#{conn.method} #{conn.request_path}"
            },
            conn,
            "medium"
          )

          unauthorized_response(conn, "Insufficient permissions")
        end
    end
  end

  defp get_required_permissions(conn, opts) do
    # Check if permissions are explicitly provided in opts
    case Keyword.get(opts, :permissions) do
      nil ->
        # Auto-detect permissions based on endpoint
        RBAC.get_endpoint_permissions(conn.method, conn.request_path)

      permissions ->
        permissions
    end
  end

  defp check_permissions(user, required_permissions, conn) do
    case required_permissions do
      [] ->
        true

      permissions when is_list(permissions) ->
        RBAC.has_all_permissions?(user, permissions)

      {resource, action} ->
        RBAC.has_permission?(user, resource, action)

      permission_func when is_function(permission_func) ->
        permission_func.(user, conn)
    end
  end

  defp unauthorized_response(conn, message) do
    conn
    |> put_status(403)
    |> put_resp_content_type("application/json")
    |> send_resp(
      403,
      Jason.encode!(%{
        error: "Forbidden",
        message: message
      })
    )
    |> halt()
  end
end
